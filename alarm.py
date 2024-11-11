from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def create_alarm_model():
    alarm_model = BayesianNetwork(
        [
            ("Burglary", "Alarm"),
            ("Earthquake", "Alarm"),
            ("Alarm", "JohnCalls"),
            ("Alarm", "MaryCalls"),
        ]
    )

    # Defining the parameters using CPT
    cpd_burglary = TabularCPD(
        variable="Burglary", variable_card=2, values=[[0.999], [0.001]],
        state_names={"Burglary": ['no', 'yes']},
    )
    cpd_earthquake = TabularCPD(
        variable="Earthquake", variable_card=2, values=[[0.998], [0.002]],
        state_names={"Earthquake": ["no", "yes"]},
    )
    cpd_alarm = TabularCPD(
        variable="Alarm",
        variable_card=2,
        values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
        evidence=["Burglary", "Earthquake"],
        evidence_card=[2, 2],
        state_names={"Burglary": ['no', 'yes'], "Earthquake": ['no', 'yes'], 'Alarm': ['yes', 'no']},
    )
    cpd_johncalls = TabularCPD(
        variable="JohnCalls",
        variable_card=2,
        values=[[0.95, 0.1], [0.05, 0.9]],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={"Alarm": ['yes', 'no'], "JohnCalls": ['yes', 'no']},
    )
    cpd_marycalls = TabularCPD(
        variable="MaryCalls",
        variable_card=2,
        values=[[0.1, 0.7], [0.9, 0.3]],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={"Alarm": ['yes', 'no'], "MaryCalls": ['yes', 'no']},
    )

    # Associating the parameters with the model structure
    alarm_model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)
    return alarm_model

if __name__ == "__main__":
    alarm_model = create_alarm_model()
    alarm_infer = VariableElimination(alarm_model)

    # Original query: Probability of John calling given an earthquake
    print("Probability of JohnCalls given Earthquake:")
    print(alarm_infer.query(variables=["JohnCalls"], evidence={"Earthquake": "yes"}))
    q = alarm_infer.query(variables=["JohnCalls", "Earthquake"], evidence={"Burglary": "yes", "MaryCalls": "yes"})
    print(q)

    # Probability of Mary calling given that John called
    print("\nProbability of MaryCalls given JohnCalls:")
    q = alarm_infer.query(variables=["MaryCalls"], evidence={"JohnCalls": "yes"})
    print(q)

    # Probability of both John and Mary calling given Alarm
    print("\nProbability of both JohnCalls and MaryCalls given Alarm:")
    q = alarm_infer.query(variables=["MaryCalls", "JohnCalls"], evidence={"Alarm": "yes"})
    print(q)

    # Probability of Alarm, given that Mary called
    print("\nProbability of Alarm given MaryCalls and JohnCalls:")
    q = alarm_infer.query(variables=["Alarm"], evidence={"JohnCalls": "yes", "MaryCalls": "yes"})
    print(q)