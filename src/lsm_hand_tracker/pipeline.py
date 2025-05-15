from .metadata      import generate_metadata
from .json_to_csv   import flatten_metadata_to_csv
from .cleaning      import clean_dataset
from .features      import engineer_features
# from .features     import apply_pca_smote
# from .training     import train_model
# from .prediction   import predict

def run_pipeline():
    """
    Ejecuta toda la cadena de procesamiento:
     1) Genera el JSON con landmarks y features
     2) Aplana el JSON a CSV
     3) (Ej.) Aplica PCA/SMOTE
     4) (Ej.) Entrena el modelo
     5) (Ej.) Ejecuta el flujo de inferencia
    """
    print("1) Generating metadata…")
    generate_metadata()

    print("2) Flattening metadata to CSV…")
    flatten_metadata_to_csv()

    print("3) Cleaning process…")
    clean_dataset()

    print("4) Features…")
    engineer_features()



    # Si ya tienes módulos de features, training y prediction:
    # print("3) Engineering features…")
    # apply_pca_smote()
    #
    # print("4) Training model…")
    # train_model()
    #
    # print("5) Testing/predicting…")
    # predict()

    print("✅ Pipeline complete!")

def main():
    run_pipeline()

if __name__ == "__main__":
    main()
