import pandas as pd
import numpy as np
from pathlib import Path

class Preprocess:
    def __init__(self, cols_path, cols2_path, cost_table_path, pca_data_items_path, projects_id_path, output_dir):
        self.cols_path = cols_path
        self.cols2_path = cols2_path
        self.cost_table_path = cost_table_path
        self.pca_data_items_path = pca_data_items_path
        self.projects_id_path = projects_id_path
        self.output_dir = output_dir
        self.processed_data = None

    def cost_table(self) -> pd.DataFrame:
        """
        This function takes in a file path as a string of the costs and
        a file path of the column references. If possible, calculate the total
        cost of each project and save in pandas DataFrame object.
        """
        pass
    
    def pca_data_items(self) -> pd.DataFrame:
        """
        This function takes in a file path as a string of the pca items and
        a file path of the column references. Then, it processes the pca items. 
        It save in pandas DataFrame object.
        """
        cols = pd.read_pickle(self.cols_path)
        pca_items = pd.read_csv(self.pca_data_items_path, dtype= {'id':str, 'project_id':str, 'document_spot':str, 'section_reference':str, })

        # get specific columns and projects based off of cols_path
        pca_items = pca_items[pca_items['document_spot'].isin(cols['source'].tolist()) & pca_items['project_id'].isin(self.projects_id_path)]

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

    def get_csv(self):
        """
        This function concludes the DataFrame object and writes it to a csv file.
        The output file path is specified in the output_dir attribute.
        """
        # Create a DataFrame from example_data
        cost_table = self.cost_table()
        pca_data_items = self.pca_data_items()

        # Append example_df to empty_df
        final_df = pd.concat([cost_table, pca_data_items], ignore_index=True)

        # Write final_df to a csv file
        for project_id in final_df['project_id'].unique():
            final_df[final_df['project_id'] == project_id].to_csv(Path(self.output_dir, f'{project_id}_processed_data.csv'), index=False)