import skdim

def get_estimators():
    return {
        "MLE": lambda k: skdim.id.MLE(neighborhood_based=True, n_neighbors=k),
        "TwoNN": lambda k: skdim.id.TwoNN(),
        "FisherS": lambda k: skdim.id.FisherS(project_on_sphere=False),
        "MOM": lambda k: skdim.id.MOM(),
        "TLE": lambda k: skdim.id.TLE(),
        "CorrInt": lambda k: skdim.id.CorrInt(),
        "DANCo": lambda k: skdim.id.DANCo(k=k),
        "ESS": lambda k: skdim.id.ESS(),
        "MiND_ML": lambda k: skdim.id.MiND_ML(ver="ML"),
        "MiND_KL": lambda k: skdim.id.MiND_ML(ver="KL"),
        "MADA": lambda k: skdim.id.MADA(),
    }


def estimate_id(embed, methods, k_values):
    results = []
    estimators = get_estimators()

    for method in methods:
        for k in k_values:
            try:
                print("METHOD ID", method)
                estimator = estimators[method](k)
                estimator.fit(embed)

                if hasattr(estimator, "dimension_"):
                    val = float(estimator.dimension_)
                elif hasattr(estimator, "dimension_pw_"):
                    val = float(np.nanmean(estimator.dimension_pw_))
                else:
                    raise RuntimeError(f"No dimension attribute for {method}")

                print("ID", val)
                results.append(
                    {
                        "method": method,
                        "k": k,
                        "global_id": val,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "method": method,
                        "k": k,
                        "global_id": np.nan,
                        "error": str(e),
                    }
                )

    return results


