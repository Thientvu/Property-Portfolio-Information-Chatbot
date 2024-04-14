import pandas as pd
import os

# Load datasets
projects_df = pd.read_csv('pca_data/chatbot_projects.csv')
pca_items_df = pd.read_csv('pca_data/chatbot_pca_data_items.csv')
cost_tables_df = pd.read_csv('pca_data/chatbot_cost_tables_ts.csv')

# Data joining
combined_df = pd.merge(projects_df, pca_items_df, left_on='id', right_on='project_id', how='inner')
final_df = pd.merge(combined_df, cost_tables_df, on='project_id', how='inner')

# Create directories and save each project's data into its own CSV
for (client, portfolio), group in final_df.groupby(['client', 'portfolio']):
    directory = f"{client}_{portfolio}".replace(" ", "_").replace("/", "_")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for project_id, project_group in group.groupby('project_id'):
        filename = f"{directory}/project_{project_id}.csv"
        project_group.to_csv(filename, index=False)
