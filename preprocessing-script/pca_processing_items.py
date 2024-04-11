import pandas as pd
import numpy as np
from pathlib import Path

def process_pca_items(file_path: str, cols_path: str, projects_path: str) -> pd.DataFrame:
    """
    This function takes in a file path as a string of the pca items and
    a file path of the column references. Then, it processes the pca items. 
    It returns a pandas DataFrame object.
    """
    # read file paths
    project_id_lst = pd.read_csv(projects_path, dtype={'id': str})['id'].unique().tolist()
    cols = pd.read_pickle(cols_path)
    pca_items = pd.read_csv(file_path, dtype= {'id':str, 'project_id':str, 'document_spot':str, 'section_reference':str, })

    # get specific columns and projects based off of cols_path
    pca_items = pca_items[pca_items['document_spot'].isin(cols['source'].tolist()) & pca_items['project_id'].isin(project_id_lst)]

    # create a new document_data column where document_text is null (then include the_data and document_text)
    pca_items['document_data'] = np.where(pca_items['document_text'].isnull(), pca_items['the_data'], pca_items['document_text'])

    # drop unneeded columns
    pca_items = pca_items.drop(['document_text', 'the_data', 'id'], axis=1)
    
    pca_items = pca_items.dropna(subset = ['document_data'])

    # only use specific sections based off of cols_path, section 1, and section 2.3
    pca_items = pca_items[pca_items['section_reference'].isin(cols['section_reference'].tolist())
                          | pca_items['section_reference'].str.startswith('1')
                          | pca_items['section_reference'].str.startswith('2.3')]

    # remove any sections that start with pca_reserve
    pca_items = pca_items[~pca_items['document_spot'].str.startswith('pca_reserve')]

    # delete section_reference
    del pca_items['section_reference']

    # rename document_spot to source
    pca_items.rename(columns={'document_spot': 'source'}, inplace=True)

    # merge with data in cols_path
    pca_items = pca_items.merge(cols, on='source', how='left')

    # delete source
    del pca_items['source']

    # create new docu_txt column that looks at the target and document_data column
    # and adds them together
    pca_items['docu_txt'] = pca_items['target'] + ' is ' + pca_items['document_data']

    # group the DataFrame object by project_id and section_reference and join the strings by a comma
    pca_items_gb = pca_items.groupby(['project_id', 'section_reference'])['docu_txt'].apply(lambda x: ', '.join(x)).reset_index()

    # set cost_related column to false
    pca_items_gb['cost_related'] = False

    return pca_items_gb
    
if __name__ == '__main__':
    table_b = process_pca_items('../dat/raw-2/chatbot_pca_data_items.csv', 'col_reference.pkl', '../dat/raw-2/chatbot_projects.csv')

    print(table_b.head())
    