Place trained model files here (see backend/models/loader.py).

Expected files (examples):
- Maharastra_model_bundle.pkl
- patient_flow_model.pkl  (optional; enables /predict/patient-flow trained path)
- readmission_model.pkl   (optional; enables /predict/readmission-risk trained path)

If .pkl files are missing, prediction endpoints still respond using heuristic fallbacks.
