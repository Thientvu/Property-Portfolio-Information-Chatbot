import pandas as pd
import numpy as np
from pathlib import Path


class Preprocessor:
    def __init__(self, portfolio_id, cols, cols2, cost_tables, data_items, projects):
        self.portfolio_id = portfolio_id
        self.cols = pd.read_pickle(cols).head()
        self.cols2 = pd.read_pickle(cols2).head()
        self.cost_tables = cost_tables
        self.data_items = data_items
        self.projects = projects

        self.project_id_lst = self.get_portfolio_project_ids()
        self.ts_st_gb = None
        self.ts_lt_gb = None

    def get_portfolio_project_ids(self):
        projects_df = pd.read_csv(projects, dtype={'id': str})
        portfoilo_df = projects_df[projects_df['portfolio_id'] == self.portfolio_id]
        return portfoilo_df['id'].unique().tolist()

    def cost_tables_processor(self):
        ts = pd.read_csv(self.cost_tables,
                         dtype={'a_column': str, 'id': str, 'a_column': str, 'project_id': str}
                         # parse_dates=['created_at', 'updated_at']
                         )

        ts = ts[ts['project_id'].isin(self.project_id_lst)]
        ts_st = ts[ts['name'] == 'Short-term'].copy()
        ts_lt = ts[ts['name'] == 'Long-term'].copy()

        st_cols = ['project_id', 'a_column', 'b_column', 'c_column', 'd_column',
                   'e_column', 'g_column', 'h_column']
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

        ts_st['docu_txt'] = ('There is an ' + ts_st['cost_type'] + ' cost item regarding ' + ts_st[
            'cost_item_description'] + ' for total unit of ' + ts_st['number of units'].astype(str) +
                             ' with total cost of ' + ts_st['total_cost'].astype(str) + ' dollars.')
        ts_st_gb = ts_st.groupby(['project_id', 'section_reference'])['docu_txt'].apply(
            lambda x: ', '.join(x)).reset_index()

        self.ts_st_gb = ts_st_gb.head()

        lt_cols = ['project_id', 'a_column', 'b_column', 'e_column', 'f_column', 'g_column', 'n_column',
                   'j_column', 'tab_name',
                   'year_1', 'year_2', 'year_3', 'year_4', 'year_5', 'year_6', 'year_7', 'year_8', 'year_9',
                   'year_10', 'year_11', 'year_12']
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
                              }, inplace=True)

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
        self.ts_lt_gb = ts_lt.groupby(['project_id', 'section_reference'])['docu_txt'].apply(
            lambda x: ', '.join(x)).reset_index()

        ts_lt_union = pd.concat([ts_st_gb, self.ts_lt_gb], ignore_index=True)
        ts_lt_union['cost_related'] = True

        return ts_lt_union.head()

    def data_items_processor(self):
        pca_items = pd.read_csv(self.data_items,
                                dtype={'id': str, 'project_id': str, 'document_spot': str,
                                       'section_reference': str, })

        pca_items['document_data'] = np.where(pca_items['document_text'].isnull(), pca_items['the_data'],
                                              pca_items['document_text'])
        pca_items = pca_items.drop(['document_text', 'the_data', 'id'], axis=1)

        pca_items = pca_items.dropna(subset=['document_data'])
        pca_items = pca_items[pca_items['section_reference'].isin(self.cols['section_reference'].tolist())
                              | pca_items['section_reference'].str.startswith('1')
                              | pca_items['section_reference'].str.startswith('2.3')]
        pca_items = pca_items[~pca_items['document_spot'].str.startswith('pca_reserve')]

        del pca_items['section_reference']
        pca_items.rename(columns={'document_spot': 'source'}, inplace=True)

        pca_items = pca_items.merge(self.cols, on='source', how='left')

        del pca_items['source']

        pca_items['docu_txt'] = pca_items['target'] + ' is ' + pca_items['document_data']

        pca_items_gb = pca_items.groupby(['project_id', 'section_reference'])['docu_txt'].apply(
            lambda x: ', '.join(x)).reset_index()

        pca_items_gb['cost_related'] = False

        return pca_items_gb

    def merge_parts(self):
        self.cost_tables_processor()
        pca_items_gb = self.data_items_processor()

        # print('pca_items_gb')
        # print(pca_items_gb)
        #
        # print('ts_st_gb')
        # print(self.ts_st_gb)
        #
        # print('ts_lt_gb')
        # print(self.ts_lt_gb)

        df_union = pd.concat([pca_items_gb, self.ts_st_gb, self.ts_lt_gb], ignore_index=True)
        df_union = df_union.merge(self.cols2, on='section_reference', how='left')
        df_union = df_union[['project_id', 'section_reference', 'category', 'section', 'subsection',
                             'cost_related', 'docu_txt']]

        return df_union

    def export_processed_data(self):
        parent_dir = Path('chatbot_doc_export_' + str(self.portfolio_id))

        # Create the directory if it doesn't exist
        parent_dir.mkdir(parents=True, exist_ok=True)

        merge_parts = self.merge_parts()

        for project_id in self.project_id_lst:
            export_df = merge_parts[merge_parts['project_id'] == project_id].iloc[:, 1:]
            export_df.to_csv(f'{parent_dir}/{project_id}_data.csv',
                             index=False)

if __name__ == '__main__':
    cols = 'preprocessing-script/col_reference.pkl'
    cols2 = 'preprocessing-script/cols2.pkl'
    cost_tables = 'dat/raw-2/chatbot_cost_tables_ts.csv'
    data_items = 'dat/raw-2/chatbot_pca_data_items.csv'
    projects = 'dat/raw-2/chatbot_projects.csv'

    preprocess_data = Preprocessor(231, cols, cols2, cost_tables,
                                   data_items, projects)

    preprocess_data.export_processed_data()

    # print(preprocess_data.merge_parts())
    # preprocess_data.merge_parts()



