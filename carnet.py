from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

def create_car_model():
    car_model = BayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition","Starts"),
            ("Gas","Starts"),
            ("Starts","Moves")
        ]
    )

    # Defining the parameters using CPT
    from pgmpy.factors.discrete import TabularCPD

    cpd_battery = TabularCPD(
        variable="Battery", variable_card=2, values=[[0.70], [0.30]],
        state_names={"Battery":['Works',"Doesn't work"]},
    )

    cpd_gas = TabularCPD(
        variable="Gas", variable_card=2, values=[[0.40], [0.60]],
        state_names={"Gas":['Full',"Empty"]},
    )

    cpd_radio = TabularCPD(
        variable=  "Radio", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                     "Battery": ['Works',"Doesn't work"]}
    )

    cpd_ignition = TabularCPD(
        variable=  "Ignition", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                     "Battery": ['Works',"Doesn't work"]}
    )

    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
        evidence=["Ignition", "Gas"],
        evidence_card=[2, 2],
        state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
    )

    cpd_moves = TabularCPD(
        variable="Moves", variable_card=2,
        values=[[0.8, 0.01],[0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                     "Starts": ['yes', 'no'] }
    )

    # Associating the parameters with the model structure
    car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)
    return car_model

if __name__ == "__main__":
    car_model = create_car_model()
    car_infer = VariableElimination(car_model)

    # Original Query: Computing the probability of the car moving given that the radio turns on and the car starts
    print("\nProbability of the car moving given that the radio turns on and the car starts:")
    print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

    # Given that the car will not move, what is the probability that the battery is not working?
    print("\nProbability that the battery is not working given that the car will not move:")
    print(car_infer.query(variables=["Battery"],evidence={"Moves":"no"}))

    # Given that the radio is not working, what is the probability that the car will not start?
    print("\nProbability that the car will not start given that the radio is not working:")
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"}))

    # Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
    print("\nProbability of the radio working given that the battery is working and that the car has gas in it:")
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"}))

    # Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car dies not have gas in it?
    print("\nProbability of the ignition failing given that the car doesn't move and that the car dies not have gas in it:")
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"}))

    # What is the probability that the car starts if the radio works and it has gas in it?
    print("\nProbability that the car starts if the radio works and it has gas in it:")
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"turns on", "Gas":"Full"}))
