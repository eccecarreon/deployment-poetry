from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

# Preprocessing pipeline
def create_preprocessing_pipeline():

    # Select numeric and categorical columns
    num_cols = make_column_selector(dtype_include='number')
    cat_cols = make_column_selector(dtype_include='object')

    # Instantiate the transformers
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')

    # Create pipeline
    num_pipe = Pipeline([
        ('scaler', scaler),
        ('imputer', knn_imputer)
    ])

    cat_pipe = Pipeline([
        ('encoder', encoder)
    ])

    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols)
    ], remainder='drop')

    return preprocessor

# Create preprocessor object
preprocessor = create_preprocessing_pipeline()

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create sampler pipeline
def sampler_pipeline(sampler):
    return ImbPipeline([
        ('sampler', sampler)
    ])

# Preprocess and rebalance the data
def preprocess_and_rebalance_data(preprocessor, X_train, X_test, y_train):

    # Transform training data to the fitted transformer
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Create sampling pipeline
    sampler = sampler_pipeline(SMOTE(random_state=42))

    # Rebalance the data
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_transformed, y_train)

    return X_train_balanced, X_test_transformed, y_train_balanced