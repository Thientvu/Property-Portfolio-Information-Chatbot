import pandas as pd
import numpy as np
from pathlib import Path

class Preprocess:
    def __init__(self, portfolio_id, cols_path, cols2_path, cost_table_path, pca_data_items_path, projects_id_path, output_dir):
        self.cols = pd.read_pickle(cols_path)
        self.cols2 = pd.read_pickle(cols2_path)
        self.cost_table_db = pd.read_csv(cost_table_path, dtype= {'id':str, 'project_id':str, 'document_spot':str, 'section_reference':str, })
        self.pca_data_items_db = pd.read_csv(pca_data_items_path,
                         dtype={'a_column': str, 'id': str, 'a_column': str, 'project_id': str}
                         # , parse_dates=['created_at','updated_at'] # Omitting this because the table we were
                         # given don't have created_at or updated_at
                         )
        self.projects_id_db = pd.read_csv(projects_id_path, dtype={'id': str})['id'].unique().tolist()
        self.projects_id_w_portfolio_id = pd.read_csv(projects_id_path, dtype={'id': str, 'portfolio_id': str})
        self.portfolio_id = portfolio_id
        self.output_dir = output_dir
        self.processed_data = None

    def cost_table(self) -> pd.DataFrame:
        """
        This function takes in a file path as a string of the costs and
        a file path of the column references. If possible, calculate the total
        cost of each project and save in pandas DataFrame object.
        """
        project_id_lst =self.projects_id_db

        ts = self.cost_table_db

        ts = ts[ts['project_id'].isin(project_id_lst)]
        ts_st = ts[ts['name'] == 'Short-term'].copy()
        ts_lt = ts[ts['name'] == 'Long-term'].copy()

        st_cols = ['project_id', 'a_column', 'b_column', 'c_column', 'd_column', 'e_column', 'g_column', 'h_column']
        ts_st = ts_st[st_cols]

        conditions = [
            (ts_st['c_column'] == 'Immediate (0-12 months)'),
            (ts_st['c_column'] == 'Short-term (12-24 months)')
        ]

        choices = [
            ts_st['g_column'],
            ts_st['h_column']
        ]

        ts_st['total_cost'] = pd.to_numeric(np.select(conditions, choices, default=0)).astype(int)

        ts_st = ts_st.drop(['g_column', 'h_column'], axis=1)
        ts_st['number of units'] = ts_st['d_column'].astype(str) + ' ' + ts_st['e_column']
        ts_st = ts_st.drop(['d_column', 'e_column'], axis=1)
        ts_st.rename(columns={'a_column': 'section_reference',
                              'b_column': 'cost_item_description',
                              'c_column': 'cost_type',
                              }, inplace=True)

        ts_st['docu_txt'] = 'There is an ' + ts_st['cost_type'] + ' cost item regarding ' + ts_st[
            'cost_item_description'] + \
                            ' for total unit of ' + ts_st['number of units'].astype(str) + ' with total cost of ' + \
                            ts_st['total_cost'].astype(str) + ' dollars.'
        ts_st_gb = ts_st.groupby(['project_id', 'section_reference'])['docu_txt'].apply(
            lambda x: ', '.join(x)).reset_index()

        ts_st_gb.head()

        lt_cols = ['project_id', 'a_column', 'b_column', 'e_column', 'f_column', 'g_column', 'n_column', 'j_column',
                   'tab_name',
                   'year_1', 'year_2', 'year_3', 'year_4', 'year_5', 'year_6', 'year_7', 'year_8', 'year_9', 'year_10',
                   'year_11', 'year_12']

        ts_lt = ts_lt[lt_cols]
        ts_lt['number of units'] = ts_lt['f_column'].astype(str) + ' ' + ts_lt['g_column']
        ts_lt = ts_lt.drop(['f_column', 'g_column'], axis=1)
        ts_lt['e_column'] = ts_lt['e_column'].astype(int)
        ts_lt.rename(columns={'a_column': 'section_reference',
                              'b_column': 'cost_item_description',
                              'e_column': 'remaining useful life (years)',
                              'j_column': 'item cost type',
                              'n_column': 'total cost',
                              'tab_name': 'cost type',
                              },
                     inplace=True)

        ts_lt['docu_txt'] = 'There is an ' + ts_lt['item cost type'] + ' Reserve cost item regarding ' + ts_lt[
            'cost_item_description'] + \
                            ' for total unit of ' + ts_lt['number of units'] + ' that have ' + ts_lt[
                                'remaining useful life (years)'].astype(str) \
                            + ' years remaining useful life with total cost of ' + ts_lt['total cost'].astype(
            str) + ' dollars, year 1 costs $' + \
                            ts_lt['year_1'].astype(str) + ', year 2 costs $' + ts_lt['year_2'].astype(
            str) + ', year 3 costs $' + ts_lt['year_3'].astype(str) \
                            + ', year 4 costs $' + ts_lt['year_4'].astype(str) + ', year 5 costs $' + ts_lt[
                                'year_5'].astype(str) + ', year 6 costs $' + \
                            ts_lt['year_6'].astype(str) + ', year 7 costs $' + ts_lt['year_7'].astype(
            str) + ', year 8 costs $' + ts_lt['year_8'].astype(str) + \
                            ', year 9 costs $' + ts_lt['year_9'].astype(str) + ', year 10 costs $' + ts_lt[
                                'year_10'].astype(str) + ', year 11 costs $' + \
                            ts_lt['year_11'].astype(str) + ', year 12 costs $' + ts_lt['year_12'].astype(str) + '.'

        ts_lt['docu_txt'] = ts_lt['docu_txt'].replace(", year \d+ costs \$0\.0", "", regex=True)
        ts_lt = ts_lt[ts_lt['docu_txt'].notnull()]
        ts_lt_gb = ts_lt.groupby(['project_id', 'section_reference'])['docu_txt'].apply(
            lambda x: ', '.join(x)).reset_index()

        ts_lt_union = pd.concat([ts_st_gb, ts_lt_gb], ignore_index=True)
        ts_lt_union['cost_related'] = True

        return ts_lt_union
        pass
    
    def pca_data_items(self) -> pd.DataFrame:
        """
        This function takes in a file path as a string of the pca items and
        a file path of the column references. Then, it processes the pca items. 
        It save in pandas DataFrame object.
        """
        # read file paths
        project_id_lst = self.projects_id_db
        cols = self.cols
        pca_items = self.pca_data_items_db

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
    
    def clear_data(self):
        """
        With the final dataframe, check all projects that is not belong to portfolio_id, remove them
        """
        portfolio_projects = self.projects_id_w_portfolio_id[self.projects_id_w_portfolio_id['portfolio_id'] == self.portfolio_id]
        return portfolio_projects['id'].unique().tolist()

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

        # Create output directory if it does not exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        id_lst = self.clear_data()

        # Write final_df to a csv file
        project_count = 0
        for project_id in id_lst:
            project_count += 1
            final_df[final_df['project_id'] == project_id].to_csv(Path(self.output_dir, f'{project_id}_processed_data.csv'), index=False)

        return project_count

if __name__ == '__main__':
    myData = Preprocess('539', 'preprocessing-script/col_reference.pkl', 'preprocessing-script/cols2.pkl', 'dat/raw-2/chatbot_cost_tables_ts.csv', 'dat/raw-2/chatbot_pca_data_items.csv', 'dat/raw-2/chatbot_projects.csv', 'output')
    print(myData.get_csv())