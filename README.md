# A Lightweight Spatio-Temporal Neural Network with Sampling-based Time Series Decomposition for Traffic Forecasting

---

## 📘 Examples

Our LSTNN model is implemented based on the [BasicTS](https://github.com/GestaltCogTeam/BasicTS) benchmark library.  
Readers can access the specific usage of BasicTS through the following link:

👉 [https://github.com/GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS)

---

## 🧪 Reproducing Built-in Models

You can reproduce these models by running the following command:

```bash
python examples/run.py -c examples/LSTNN/LSTNN_${DATASET_NAME}.py --gpus '0'
