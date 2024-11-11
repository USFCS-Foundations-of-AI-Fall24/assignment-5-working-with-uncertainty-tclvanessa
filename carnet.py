from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def create_car_model():
    car_model = BayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition", "Starts"),
            ("Gas", "Starts"),
            ("Starts", "Moves"),
            ("KeyPresent", "Starts")
        ]
    )

    # Defining the parameters using CPT
    cpd_battery = TabularCPD(
        variable="Battery", variable_card=2, values=[[0.70], [0.30]],
        state_names={"Battery": ['Works', "Doesn't work"]},
    )

    cpd_gas = TabularCPD(
        variable="Gas", variable_card=2, values=[[0.40], [0.60]],
        state_names={"Gas": ['Full', "Empty"]},
    )

    cpd_radio = TabularCPD(
        variable="Radio", variable_card=2,
        values=[[0.75, 0.01], [0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                     "Battery": ['Works', "Doesn't work"]}
    )

    cpd_ignition = TabularCPD(
        variable="Ignition", variable_card=2,
        values=[[0.75, 0.01], [0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                     "Battery": ['Works', "Doesn't work"]}
    )

    cpd_key_present = TabularCPD(
        variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
        state_names={"KeyPresent": ["yes", "no"]}
    )

    # Updated CPD for Starts with KeyPresent as an additional dependency
    cpd_starts = TabularCPD(
        variable="Starts", variable_card=2,
        values=[
            [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # Starts = yes
            [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]   # Starts = no
        ],
        evidence=["Ignition", "Gas", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"], "Gas": ['Full', "Empty"], "KeyPresent": ["yes", "no"]}
    )

    cpd_moves = TabularCPD(
        variable="Moves", variable_card=2,
        values=[[0.8, 0.01], [0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                     "Starts": ['yes', 'no']}
    )

    # Associating the parameters with the model structure
    car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key_present)
    return car_model

if __name__ == "__main__":
    car_model = create_car_model()
    car_infer = VariableElimination(car_model)

    # Original Query: Probability of the car moving given that the radio turns on and the car starts
    print("\nProbability of the car moving given that the radio turns on and the car starts:")
    print(car_infer.query(variables=["Moves"], evidence={"Radio": "turns on", "Starts": "yes"}))

    # Probability that the battery is not working given that the car will not move
    print("\nProbability that the battery is not working given that the car will not move:")
    print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))

    # Probability that the car will not start given that the radio is not working
    print("\nProbability that the car will not start given that the radio is not working:")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))

    # Probability of the radio working given that the battery is working and the car has gas
    print("\nProbability of the radio working given that the battery is working and the car has gas:")
    print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"}))

    # Probability of ignition failing given that the car doesn't move and the car does not have gas
    print("\nProbability of the ignition failing given that the car doesn't move and it has no gas:")
    print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"}))

    # Probability that the car starts if the radio works and it has gas
    print("\nProbability that the car starts if the radio works and it has gas:")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))

    # Probability that the key is not present given that the car does not move
    print("\nProbability that the key is not present given that the car does not move:")
    print(car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"}))
