'''
https://pgmpy.org/examples/Earthquake.html
'''
# Importing Library
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Defining network structure

alarm_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.999], [0.001]]
)
cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.998], [0.002]]
)
cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
)
cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],
)
cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.1, 0.7], [0.9, 0.3]],
    evidence=["Alarm"],
    evidence_card=[2],
)

# Associating the parameters with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls
)


# Checking if the cpds are valid for the model
print(alarm_model.check_model())

# Viewing nodes of the model
print(alarm_model.nodes())

# Viewing edges of the model
print(alarm_model.edges())

# Checking independcies of a node
print(alarm_model.local_independencies("Burglary"))

# Listing all Independencies
print(alarm_model.get_independencies())


# https://pgmpy.org/examples/Inference%20in%20Discrete%20Bayesian%20Networks.html
# Initializing the VariableElimination class
from pgmpy.inference import VariableElimination
alarm_infer = VariableElimination(alarm_model)


# Computing the probability of bronc given smoke=no.
#q = alarm_infer.query(variables=["Alarm"], evidence={"Burglary": 1})
#q = alarm_infer.query(variables=["Alarm", "Burglary"], evidence={"MaryCalls":0, "JohnCalls":0}    )
q = alarm_infer.query(variables=["Alarm","JohnCalls"], evidence={"Burglary":1})
#q = alarm_infer.query(variables=["Burglary"])

print(q)