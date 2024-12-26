import unittest
import joblib
import os
import pandas as pd

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        """Configuration avant chaque test."""
        # Définir les chemins du modèle et du scaler
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(current_directory, "..", "mep_dash","backend", "model.joblib")
    
    def test_model_exists(self):
        """Test si le modèle a bien été sauvegardé."""
        self.assertTrue(os.path.exists(self.model_path), "Le modèle n'a pas été sauvegardé correctement.")
    
    def test_model_prediction(self):
        """Test si le modèle peut prédire avec des données du fichier CSV."""
        # Charger le modèle
        model = joblib.load(self.model_path)

        # Charger les données du fichier CSV, retirer la colonne SK_ID_CURR
        current_directory = os.path.dirname(os.path.realpath(__file__))
        X_sample = pd.read_csv(os.path.join(current_directory, "..", "mep_dash","backend", "df_nouveaux_clients.csv")).drop(columns=["SK_ID_CURR"])

        # Test simple de prédiction
        predictions = model.predict(X_sample)
    
        # Vérifier que le modèle renvoie le bon nombre de prédictions
        self.assertEqual(len(predictions), len(X_sample), "La prédiction du modèle a échoué.")

if __name__ == '__main__':
    unittest.main()


