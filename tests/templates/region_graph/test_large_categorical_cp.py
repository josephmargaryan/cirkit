import functools
import math
import random

import numpy as np
import pytest
import torch

from cirkit.pipeline import compile as compile_circuit
from cirkit.symbolic.layers import CategoricalLayer
from cirkit.symbolic.parameters import mixing_weight_factory
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.templates.utils import Parameterization, parameterization_to_factory


SEED = 17
CATEGORIES = 22
BATCH = 256
WIDTH = 4
THRESHOLD = 1e-7


def _build_circuit(num_features: int) -> torch.nn.Module:
    # CirKit's pytest fixture changes the default dtype to float64.
    # The original reproducer initializes the circuit in float32.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        region_graph = RandomBinaryTree(
            num_features,
            num_repetitions=1,
            seed=SEED,
        )

        def input_factory(scope, num_units):
            return CategoricalLayer(
                scope=scope,
                num_categories=CATEGORIES,
                num_output_units=num_units,
            )

        sum_param = Parameterization(
            activation="softmax",
            initialization="normal",
        )
        sum_weight_factory = parameterization_to_factory(sum_param)

        nary_sum_weight_factory = functools.partial(
            mixing_weight_factory,
            param_factory=sum_weight_factory,
        )

        symbolic = region_graph.build_circuit(
            input_factory=input_factory,
            sum_weight_factory=sum_weight_factory,
            nary_sum_weight_factory=nary_sum_weight_factory,
            num_input_units=WIDTH,
            num_sum_units=WIDTH,
            sum_product="cp",
        )

        circuit = compile_circuit(symbolic)
        circuit = circuit.to(
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    finally:
        torch.set_default_dtype(previous_dtype)

    return circuit


def _gradient_norm(circuit: torch.nn.Module) -> float:
    grad_sq = 0.0

    for parameter in circuit.parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().double()

        if bool(torch.isfinite(gradient).all()):
            norm = float(torch.linalg.vector_norm(gradient).cpu())
            grad_sq += norm * norm

    return math.sqrt(grad_sq)


@pytest.mark.parametrize(
    "num_features",
    [
        pytest.param(224, id="D224"),
        pytest.param(240, id="D240"),
        pytest.param(263, id="D263"),
    ],
)
def test_large_categorical_cp_does_not_degenerate(num_features: int) -> None:
    circuit = _build_circuit(num_features)

    generator = torch.Generator().manual_seed(SEED + 1)
    x = torch.randint(
        0,
        CATEGORIES,
        (BATCH, num_features),
        generator=generator,
        dtype=torch.long,
    )

    circuit.eval()

    with torch.no_grad():
        scores = circuit(x).reshape(BATCH, -1)[:, 0]

    score_std = float(
        scores.double().std(unbiased=False).cpu()
    )

    circuit.train()
    circuit.zero_grad(set_to_none=True)

    # CirKit's pytest fixture globally disables autograd, so explicitly
    # enable it for the training pass used by this regression.
    with torch.enable_grad():
        scores_train = circuit(x).reshape(BATCH, -1)[:, 0]
        loss = -scores_train.mean()
        loss.backward()

    gradient_norm = _gradient_norm(circuit)

    diagnostics = (
        f"D={num_features}, "
        f"score_std={score_std:.3e}, "
        f"gradient_norm={gradient_norm:.3e}"
    )

    assert score_std > THRESHOLD, diagnostics
    assert gradient_norm > THRESHOLD, diagnostics
