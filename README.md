# Measurement and Verification tool

## Purpose of the Tool

The tool aims to evaluate the energy savings achieved following the implementation of an **Energy Conservation Measure (ECM)**. An ECM may involve actions related to building energy retrofits, such as:

- Window replacement
- Installation of thermal insulation
- Modification of opaque/transparent surfaces and internal spaces
- Improvement of the roof's thermal properties
- Change in lighting type and/or systems

It can also include actions aimed at improving indoor comfort conditions and energy consumption through the installation of **automated control systems**.

## Reference Framework: IPMVP

The energy savings assessments are based on the **IPMVP (International Performance Measurement and Verification Protocol)**, which is the international standard for the measurement and verification of energy savings.

The tool relies on the development of **analytical models** to estimate what energy consumption would have been **in the absence of efficiency interventions**, i.e., to build the so-called **reference scenario or baseline**.

---

### 📌 What is an Analytical Model?

An analytical model is a **mathematical or statistical representation** that describes the relationship between energy consumption and independent variables (such as external temperature, occupancy, operating hours, etc.). This model is calibrated using **historical data** collected before the intervention.

---

### 🎯 Role of the Analytical Model in IPMVP

In the context of IPMVP, analytical models are commonly used in **Option C – Whole Facility Analysis**. The approach consists of:

- Building a **statistical model** (e.g., linear regression, polynomial regression, machine learning models, etc.) of energy consumption during the **baseline period** (pre-intervention).
- After the intervention, using **input data** (e.g., external or internal temperature, etc.) from the **reporting period** to simulate what energy consumption would have been according to the model.
- The **difference** between the simulated consumption (baseline) and the actual consumption represents the **energy savings**.

---

### ✅ Why Were Analytical Models Chosen?

- They allow for **continuous evaluation** of savings, even in the absence of conditions identical to the pre-intervention period.
- They are ideal for **complex systems** where many variables influence energy use.
- They handle **environmental fluctuations** (e.g., seasons, weather) more precisely.

---

### 🛠 Implementation Details

Once the dataset (cleaned) is defined, along with the input variables, output variable, and time periods (baseline — divided into test and validation — and reporting period), the tool attempts to generate analytical models using algorithms such as:

- **Polynomial Regression**
- **Random Forest**
- **LightGBM**
- **Neural Networks (LSTM)**

Starting from the simplest model, statistical parameters are evaluated. A model is considered valid only if it exceeds certain performance thresholds.

---

### ⏳ Use of Lagged Inputs

Among the input variables, **past effects** (lagged inputs) are also considered in relation to the current output (also called the target). This is based on the assumption that the **effect of certain variables does not manifest instantly**, but **after a delay** (lag). For example, the impact of outdoor temperature on energy consumption may be delayed due to thermal inertia.

###  ✅ Model evaluation
For the evaluation and selection of the analytical model, refer to the details described within the following paper:
[Building performance evaluation through a novel feature selection algorithm for automated arx model identification procedures](https://www.sciencedirect.com/science/article/abs/pii/S0378778816318291)


### 📊 Example of Pipeline

Within the `main()` file, the pipeline to be executed is defined.

In the example shown, the goal is to assess how the use of a **home automation system** with an **adaptive algorithm** can reduce energy consumption in a building by managing **indoor temperatures** during working hours, nighttime, and distinguishing between **weekdays and holidays**.

### Approach Overview

The following steps are implemented:

- **Generation of an analytical model** for indoor temperatures during the **baseline period**.
- **Simulation of the indoor temperature model** during the **reporting period**.
- **Creation of an analytical model** for the **thermal energy consumption** of the building, which also takes **indoor temperatures** into account.
- **Simulation of the thermal energy consumption model** during the **reporting period**, using the **simulated indoor temperature data** as input. These temperatures represent the values that would have occurred in the building **in the absence of automation**.
- **Evaluation of energy savings** by calculating the difference between the **actual data** and the **model-simulated data**.

### Report Generation

Inside the report folder, it is possible to generate an HTML report with KPIs.
Modify all inputs in the `generate_report_model.py` file to generate the report for your specific case.

## Contributing and support 
Bug reports/Questions If you encounter a bug, kindly create a GitLab issue detailing the bug. Please provide steps to reproduce the issue and ideally, include a snippet of code that triggers the bug. If the bug results in an error, include the traceback. If it leads to unexpected behavior, specify the expected behavior.

Code contributions We welcome and deeply appreciate contributions! Every contribution, no matter how small, makes a difference. Click here to find out more about contributing to the project.

# License
Free software: MIT license

# Contacts
For any question or support, please contact: 
- Daniele Antonucci daniele.antonucci@eurac.edu 
- Olga Somova olga.somova@eurac.edu

## 💙 Acknowledgements
This work was carried out within European projects: 

<p align="center">
  <img src="M_V/assets/moderate_logo.png" alt="moderate" style="width: 100%;">
</p>

Moderate - Horizon Europe research and innovation programme under grant agreement No 101069834, with the aim of contributing to the development of open products useful for defining plausible scenarios for the decarbonization of the built environment