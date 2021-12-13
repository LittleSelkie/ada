import pandas as pd
from openpyxl import *

def create_report(table, table_name):
    excel_path = 'excel_report.xlsx'
    book = load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    writer.book = book
    table.to_excel(writer, sheet_name=table_name)
    writer.save()
    writer.close()
