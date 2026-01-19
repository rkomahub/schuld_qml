import pennylane as qml

dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=5)

@qml.qnode(dev)
def test():
    return qml.expval(qml.NumberOperator(0))

print(test())