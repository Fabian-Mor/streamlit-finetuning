import streamlit as st
import pickle as pkl
from plotting import *

st.sidebar.header("Settings")

dataset_options = ["CIFAR100", "CIFAR10", "CIFAR101"]
dataset = st.sidebar.selectbox("Dataset", dataset_options)
plot_options = ["PCA", "Training", "Compare", "Plane", "Projections"]
plot_type = st.sidebar.selectbox("Select Plot Type", plot_options)
lr_options = ["1e-6", "1e-5", "3e-5"]
lr = st.sidebar.selectbox("Select a learning rate", lr_options)
mode = st.sidebar.radio("Choose mode:", ("accuracy", "loss"))
epoch = st.sidebar.radio("Choose epochs", ("10", "50"))
if mode == "accuracy":
    acc_loss = "accuracy"
    acc = True
else:
    acc_loss = "loss_surface"
    acc = False

# Load and plot when a selection is made
if lr:
    try:
        if plot_type == "PCA":
            with open(f"results/{dataset}/{epoch}/pca/{lr}.pkl", "rb") as f:
                result = pkl.load(f)
            explained_variance = result["explained_variance"]
            projected = result["projected"]
            base_point = result["base_point"]
            finetuned_points = result["finetuned_points"]
            interpolated_points = result["interpolated_points"]
            losses_all = result["losses"]
            acc_all = result["acc"]
            fig = plot_PCA(explained_variance, base_point, finetuned_points,
                           interpolated_points, losses_all, acc_all, acc=acc)
            st.pyplot(fig)
        elif plot_type == "Compare":
            if epoch == "10":
                with open(f"results/{dataset}/{epoch}/pca/{lr}.pkl", "rb") as f:
                    result = pkl.load(f)
                    losses_all = result["losses"]
                    acc_all = result["acc"]
                    n = len(result["finetuned_points"])
                    losses_finetuned = losses_all[:1+n]
                    acc_finetuned = acc_all[:1+n]
                    losses_interpolated = [losses_all[0]] + losses_all[1+n:]
                    acc_interpolated = [acc_all[0]] + acc_all[1+n:]
                    fig = plot_compare(losses_finetuned, acc_finetuned,
                                 losses_interpolated, acc_interpolated, acc=acc)
                    st.pyplot(fig)
            else:
                with open(f"results/{dataset}/{epoch}/compare/{lr}.pkl", "rb") as f:
                    result = pkl.load(f)
                    losses_finetuned = result["losses finetuned"]
                    acc_finetuned = result["accs finetuned"]
                    losses_interpolated = result["losses interpolated"]
                    acc_interpolated = result["accs interpolated"]
                    fig = plot_compare(losses_finetuned, acc_finetuned,
                                       losses_interpolated, acc_interpolated, acc=acc)
                    st.pyplot(fig)
        elif plot_type == "Surface":
            with open(f'results/{dataset}/{epoch}/{lr}.pkl', 'rb') as f:
                result = pkl.load(f)
            u_vals = result['u_vals']
            v_vals = result['v_vals']
            corner_names = result['names']
            corner_labels = {
                (0, 0): corner_names[0],
                (0, len(v_vals) - 1): corner_names[1],
                (len(u_vals) - 1, 0): corner_names[2],
                (len(u_vals) - 1, len(v_vals) - 1): corner_names[3],
            }
            if acc:
                title = "Accuracy Surface"
            else:
                title = "Loss Surface"
            fig = plot_contour(
                u_vals,
                v_vals,
                result[acc_loss],
                title=title,
                corner_labels=corner_labels,
            )
            st.pyplot(fig)
        elif plot_type == "Training":
            with open(f'results/{dataset}/{epoch}/opt_inter/{lr}.pkl', 'rb') as f:
                result = pkl.load(f)
            losses = result['losses']
            accs = result['accs']
            fig = plot_training(losses, accs, acc=acc)
            st.pyplot(fig)
        elif plot_type == "Plane":
            with open(f'results/{dataset}/{epoch}/triangle/{lr}.pkl', 'rb') as f:
                result = pkl.load(f)
            coords = result['coords']
            values = result['results']
            names = result['names']
            fig = plot_triangle(coords, values, acc=acc, corner_names=names)
            st.pyplot(fig)
        elif plot_type == "Projections":
            # Make this not hardcoded in the future
            names = ["1e-6", "1e-5", "3e-5"]
            with open(f'results/{dataset}/back.pkl', 'rb') as f:
                result = pkl.load(f)

            alpha_options = [0, 0.25, 0.5, 0.75, 1]
            alphas_selected = st.multiselect("Select Alphas", alpha_options)
            alphas_selected = sorted(alphas_selected)
            for alpha in alphas_selected:
                coords = result[alpha]['coords']
                values = result[alpha]['values']
                title = f"Accuracy, alpha={alpha}" if acc else f"Loss, alpha={alpha}"
                fig = plot_triangle(coords, values, acc=acc, corner_names=names, title=title)
                st.pyplot(fig)
        elif plot_type == "Pyramid":
            names = ["1e-6", "1e-5", "3e-5"]
            with open(f'results/{dataset}/{epoch}/pyramid.pkl', 'rb') as f:
                result = pkl.load(f)

            alpha_options = [0, 0.25, 0.5, 0.75, 1]
            alphas_selected = st.multiselect("Select Alphas", alpha_options)
            alphas_selected = sorted(alphas_selected)
            for alpha in alphas_selected:
                coords = result[alpha]['coords']
                values = result[alpha]['values']
                print(alpha)
                print(values)
                title = f"Accuracy, alpha={alpha}" if acc else f"Loss, alpha={alpha}"
                fig = plot_triangle(coords, values, acc=acc, corner_names=names, title=title)
                st.pyplot(fig)


    except FileNotFoundError:
        st.error(f"No file found for {lr}. Please check your selection.")