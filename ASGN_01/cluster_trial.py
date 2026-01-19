import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def kmeans_cluster_csv(
    csv_path: str,
    cluster_cols: list[str],
    k: int,
    out_path: str = "clustered_output.csv",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simple K-means:
      - reads CSV
      - uses only cluster_cols
      - imputes missing values (median)
      - standardizes features
      - runs KMeans
      - writes CSV with a new 'cluster' column
    """
    df = pd.read_csv(csv_path)

    # take only selected cols, coerce to numeric
    X = df[cluster_cols].apply(pd.to_numeric, errors="coerce")

    # impute + scale
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_std = StandardScaler().fit_transform(X_imp)

    # fit kmeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    df["cluster"] = km.fit_predict(X_std)

    df.to_csv(out_path, index=False)
    return df


# ---- example ----
if __name__ == "__main__":
    clustered = kmeans_cluster_csv(
        csv_path="DBs/database_extended.csv",
        # cluster_cols=['edresp','medied',
        #               'ageresp',
        #               'sexresp',
        #               ],
        cluster_cols=[
                     'Be_Effective',
                      'Fix_Root_Cause',
                      'No_Side_Effect',
                      'Right_Price',
                      'Not_Too_Strong',
                      'Good_Brand',
                      'Easy_To_Find',
                      'Reduce_Gas',
                      'Long_Term_Effective',
                      'Doc_Reccomend',
                      'Sustained_Relief',
                      'Flavours'
                      ],
        # cluster_cols=['edresp','occresp','medied','qlfn_2','qlfn_1',
        #               'qlfn_3','cook_1','cook_2','cook_3','eatfreq',
        #               'travel','daystrav','ailment1','ailment2','ailment3',
        #               'ailment4','occcwe','edcwe','sec','mhi','ageresp',
        #               'sexresp','edhw','workhw','lang_1','lang_2','anydr',
        #               'chemist','family','expmedi','moub'
        #               ],
        k=5,
        out_path="your_data_with_clusters.csv",
    )
    print(clustered["cluster"].value_counts().sort_index())

