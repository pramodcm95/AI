import logging
from utilities import visualize, BestClassifier, save_model, load_model
from visualize import Visualize
from sklearn.model_selection import train_test_split
from preprocess import PreProcess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

if __name__ == '__main__':
    pre_process = PreProcess()
    # Import and pre process training data
    df, features, labels, features_scaled = pre_process.execute()

    # Data Visualization
    visualize(df)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=0)

    # Initialize visualise class instance
    vis = Visualize(features_scaled, X_train, X_test, y_train, y_test)

    vis.vis_dataset()

    # choosing the best classification model from sklearn for the current data
    model = BestClassifier(X_train, X_test, y_train, y_test)
    results = model.execute()
    save_model(results.iloc[0][4], 'best_classifier.pkl')

    # visualize classifier decision
    vis.vis_classifier(results['Estimator'][0], results['Accuracy'][0])

    # Load the best model
    model = load_model('best_classifier.pkl')
    # load and pre process inference data
    _, features_inference, _ = pre_process.load_data(train=False, inference=True)
    features_inference_scaled = pre_process.preprocess_data(features_inference, train=False, inference=True)
    # infer from best model
    labels_pred = model.predict(features_inference_scaled)

    # visualize inference data boundary
    vis.vis_inference(features_inference_scaled, labels_pred)