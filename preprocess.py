import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self, col_reference, cols2, cost_table, pca_data, projects, output_dir):
        self.col_reference = col_reference
        self.cols2 = cols2
        self.cost_table = cost_table
        self.pca_data = pca_data
        self.projects = projects
        self.output_dir = output_dir

    def cost_table(self, file_path: str, projects_path: str) -> pd.DataFrame:
        project_id_lst = pd.read_csv(projects_path, dtype={'id': str})['id'].unique().tolist()

        ts = pd.read_csv(file_path,
                         dtype={'a_column': str, 'id': str, 'a_column': str, 'project_id': str}
                         # , parse_dates=['created_at','updated_at'] # Omitting this because the table we were
                         # given don't have created_at or updated_at
                         )

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

    def pca_data_items(self):
        pass

    def get_csv(self):
        pass

if __name__ == '__main__':
    pass