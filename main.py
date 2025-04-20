import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

mlflow.set_tracking_uri("https://dagshub.com/api/v1/repo-buckets/s3/amankumar536894")
os.environ['MLFLOW_TRACKING_USERNAME'] = "amankumar536894"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "1720ea79abe1eb3c1178b7a84b68db0afad0bd31"

# Set MLflow tracking (optional)
# mlflow.set_experiment("Iris_ANN_Autolog_Experiment")

# Enable autologging
mlflow.tensorflow.autolog()

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = tf.keras.utils.to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# MLflow automatically logs everything in this run
with mlflow.start_run() as run:
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # You can still register the model manually if needed
    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="IrisKerasANNModelAuto"
    )
