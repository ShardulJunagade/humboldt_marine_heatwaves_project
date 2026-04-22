import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    spatial_path = os.path.join(PROC_DIR, "spatial_monthly_sst.csv")
    drivers_path = os.path.join(PROC_DIR, "drivers_monthly.csv")

    if not os.path.exists(spatial_path):
        raise FileNotFoundError("spatial_monthly_sst.csv not found. Run build_spatial_monthly_sst.py first.")

    spatial = pd.read_csv(spatial_path, parse_dates=["date"])
    spatial["month"] = spatial["date"].dt.month

    clim = (
        spatial.groupby(["month", "latitude", "longitude"], as_index=False)["sst"]
        .mean()
        .rename(columns={"sst": "sst_clim"})
    )
    spatial = spatial.merge(clim, on=["month", "latitude", "longitude"], how="left")
    spatial["anom"] = spatial["sst"] - spatial["sst_clim"]

    mat = spatial.pivot_table(index="date", columns=["latitude", "longitude"], values="anom")
    mat = mat.sort_index().interpolate(limit_direction="both").fillna(0)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(mat.values)

    pc_df = pd.DataFrame({"date": mat.index, "pc1": pcs[:, 0], "pc2": pcs[:, 1]})
    comp1 = pd.Series(pca.components_[0], index=mat.columns).reset_index(name="loading")
    comp2 = pd.Series(pca.components_[1], index=mat.columns).reset_index(name="loading")
    comp1.columns = ["latitude", "longitude", "loading"]
    comp2.columns = ["latitude", "longitude", "loading"]

    explained = pca.explained_variance_ratio_
    expl_df = pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": explained[:2]})

    pc_df.to_csv(os.path.join(PROC_DIR, "pca_pcs_monthly.csv"), index=False)
    comp1.to_csv(os.path.join(PROC_DIR, "pca_component1_loadings.csv"), index=False)
    comp2.to_csv(os.path.join(PROC_DIR, "pca_component2_loadings.csv"), index=False)
    expl_df.to_csv(os.path.join(PROC_DIR, "pca_explained_variance.csv"), index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(pc_df["date"], pc_df["pc1"], label="PC1", color="#1f77b4")
    plt.plot(pc_df["date"], pc_df["pc2"], label="PC2", color="#ff7f0e", alpha=0.8)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Leading PCA Modes of Humboldt SST Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Principal Component Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_modes_timeseries.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, comp, title in zip(
        axes,
        [comp1, comp2],
        ["Component 1 Spatial Loadings", "Component 2 Spatial Loadings"],
    ):
        piv = comp.pivot(index="latitude", columns="longitude", values="loading")
        im = ax.imshow(
            piv.values,
            origin="lower",
            aspect="auto",
            extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()],
            cmap="coolwarm",
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_spatial_loadings.png"), dpi=300)
    plt.close()

    if os.path.exists(drivers_path):
        drivers = pd.read_csv(drivers_path, parse_dates=["date"])
        if "oni" in drivers.columns:
            joined = pc_df.merge(drivers[["date", "oni"]], on="date", how="inner").dropna()
            if not joined.empty:
                r = joined["pc1"].corr(joined["oni"])
                m, b = np.polyfit(joined["oni"], joined["pc1"], 1)
                x = np.linspace(joined["oni"].min(), joined["oni"].max(), 100)

                plt.figure(figsize=(6, 5))
                plt.scatter(joined["oni"], joined["pc1"], alpha=0.7)
                plt.plot(x, m * x + b, color="red", linewidth=2, label=f"r={r:.2f}")
                plt.title("PC1 vs ONI")
                plt.xlabel("ONI")
                plt.ylabel("PC1")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, "pc1_vs_oni.png"), dpi=300)
                plt.close()

                joined.to_csv(os.path.join(PROC_DIR, "pc1_oni_merged.csv"), index=False)

    print(f"PCA done. Explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}")


if __name__ == "__main__":
    main()
