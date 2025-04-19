import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

# -------------------- Custom CSS for Biomedical-Tech Theme --------------------
custom_css = """
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
    body {
        font-family: 'Inter', sans-serif;
        background-color: #E6F0FA;
        color: #1A202C;
    }
    .stButton>button {
        background: linear-gradient(to right, #ED8936, #F6AD55);
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #F6AD55, #ED8936);
        transform: scale(1.05);
    }
    .card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 24px;
        border: 1px solid #2B6CB0;
    }
    .header {
        background: linear-gradient(to right, #2B6CB0, #38B2AC);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        border-bottom: 4px solid #ED8936;
    }
    .section-title {
        color: #2B6CB0;
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 16px;
    }
    .sidebar .stSlider label {
        color: #2B6CB0;
        font-weight: 500;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------- Genetic Algorithm Core --------------------
def initialize_population(n_features, pop_size):
    population = []
    for _ in range(pop_size):
        individual = [0] * n_features
        while sum(individual) == 0:
            individual = np.random.choice([0, 1], size=n_features).tolist()
        population.append(individual)
    print(f"Initialized population with {pop_size} individuals.")
    return population

def fitness(individual, X, y):
    if sum(individual) == 0:
        print("Warning: Skipping individual with 0 selected features.")
        return 0
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    X_selected = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)
    return score

def selection(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual, mutation_rate):
    return [bit if random.random() > mutation_rate else 1 - bit for bit in individual]

def genetic_algorithm(X, y, pop_size=20, generations=10, mutation_rate=0.01):
    n_features = X.shape[1]
    population = initialize_population(n_features, pop_size)
    best_individual = None
    best_score = 0
    fitness_progress = []

    for gen in range(generations):
        fitnesses = [fitness(ind, X, y) for ind in population]
        best_gen_score = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Generation {gen+1}/{generations} | Best: {best_gen_score:.4f} | Avg: {avg_fitness:.4f}")

        fitness_progress.append(best_gen_score)

        if best_gen_score > best_score:
            best_score = best_gen_score
            best_individual = population[np.argmax(fitnesses)]

        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population

    return best_individual, best_score, fitness_progress

# -------------------- Visualization --------------------
def plot_feature_importance(X, y, selected_features_indices, feature_names):
    model = RandomForestClassifier()
    X_selected = X[:, selected_features_indices]
    model.fit(X_selected, y)
    importance = model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(selected_features_indices))
    ax.barh(y_pos, importance, align='center', color='#2B6CB0')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in selected_features_indices], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance Ranking', fontsize=14, pad=15)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor('#E6F0FA')
    fig.patch.set_facecolor('#E6F0FA')
    
    return fig

def plot_progress(progress):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(progress, color="#38B2AC", marker='o', linewidth=2)
    ax.set_title("Accuracy Over Generations", fontsize=14, pad=15)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor('#E6F0FA')
    fig.patch.set_facecolor('#E6F0FA')
    return fig

# -------------------- Streamlit App --------------------
def main():
    st.markdown('<div class="header"><h1 class="text-3xl font-bold">NeuroGenis: Genetic Feature Selector</h1><p class="text-lg">Optimize feature subsets using a genetic algorithm by Jacqueline Chiazor</p></div>', unsafe_allow_html=True)

    # Sidebar for settings
    with st.sidebar:
        st.markdown('<div class="card"><h2 class="section-title">Algorithm Settings</h2></div>', unsafe_allow_html=True)
        pop_size = st.slider("Population Size", 10, 100, 20, step=10)
        generations = st.slider("Generations", 5, 100, 20, step=5)
        mutation_rate = st.slider("Mutation Rate", 0.0, 0.5, 0.01, step=0.01)

    # Main content
    with st.container():
        st.markdown('<div class="card"><h2 class="section-title">Upload Dataset</h2></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.markdown('<div class="card"><h3 class="section-title">Dataset Preview</h3></div>', unsafe_allow_html=True)
            st.dataframe(df.head())
        else:
            st.warning("Please upload your dataset to begin.")
            return

        if df.shape[1] < 2:
            st.warning("Dataset must have at least 2 columns")
            return

        # EXPLICIT TARGET COLUMN
        target_col = "diagnosis"
        if target_col not in df.columns:
            st.error(f"Target column '{target_col}' not found in your dataset.")
            return

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        y = y.astype(int)

        if st.button("Run Genetic Algorithm"):
            with st.spinner("Optimizing feature subset..."):
                best_ind, best_score, progress = genetic_algorithm(
                    X, y, pop_size, generations, mutation_rate
                )

            st.markdown('<div class="card"><h3 class="section-title">Optimization Results</h3></div>', unsafe_allow_html=True)
            st.success(f"Best Accuracy: {best_score:.4f}")

            selected_features = [df.columns[i] for i, bit in enumerate(best_ind) if bit == 1]
            selected_indices = [i for i, bit in enumerate(best_ind) if bit == 1]

            st.markdown('<div class="card"><h3 class="section-title">Selected Features</h3></div>', unsafe_allow_html=True)
            for feat in selected_features:
                st.markdown(f"- {feat}")
            st.markdown(f"**{len(selected_features)} selected out of {X.shape[1]} features**")

            st.markdown('<div class="card"><h3 class="section-title">Feature Importance</h3></div>', unsafe_allow_html=True)
            fig = plot_feature_importance(X, y, selected_indices, df.columns)
            st.pyplot(fig)

            st.markdown('<div class="card"><h3 class="section-title">Progress Over Generations</h3></div>', unsafe_allow_html=True)
            fig2 = plot_progress(progress)
            st.pyplot(fig2)

            # Download optimized dataset
            st.markdown('<div class="card"><h3 class="section-title">Download Results</h3></div>', unsafe_allow_html=True)
            selected_df = df[selected_features + [target_col]]
            csv = selected_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Selected Features", data=csv, file_name="optimized_features.csv", key="download-btn")

if __name__ == "__main__":
    main()