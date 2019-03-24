import xlsxwriter
import pandas as pd
import random

urgent_tickets = []

def get_urgents(filename):
    data = pd.read_excel(filename)
    for row in data.values:
        ticket = {
            "id": str(row[0]),
            "title": str(row[1]),
            "body": str(row[2]),
            "tag": str(row[3])
        }

        if row[3] == 0 and not ("Additional Details") in ticket["title"]:
            ticket["tag"] = "not"
            urgent_tickets.append(ticket)


def shuffle(filename):
    data = pd.read_excel(filename)
    tickets = []

    for row in data.values:
        ticket = {
            "id": str(row[0]),
            "title": str(row[1]),
            "body": str(row[2]),
            "tag": str(row[3])
        }
        if ticket['tag'] == '1':
            ticket["tag"] = "urgent"
        else:
            ticket["tag"] = "not"

        tickets.append(ticket)

    print(len(tickets))
    random.shuffle(tickets)

    return tickets

def export_excel(tickets, filename):
    workbook = xlsxwriter.Workbook('./temp/datasets/' + filename + '.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, 'ID')
    worksheet.write(0, 1, 'Title')
    worksheet.write(0, 2, 'Body')
    worksheet.write(0, 3, 'Tag')

    row = 1
    for ticket in tickets:
        worksheet.write(row, 0, ticket['id'])
        worksheet.write(row, 1, ticket['title'])
        worksheet.write(row, 2, ticket['body'])

        if 'urgent' in ticket['tag']:
            worksheet.write(row, 3, 1)
        else:
            worksheet.write(row, 3, 0)

        row += 1

    workbook.close()

# for i in range(1, 9):
#     filename = "./temp/datasets/dataset-" + str(i) + ".xlsx"
#     get_urgents(filename)
#     print(len(urgent_tickets))
tick = shuffle("./temp/datasets/dataset_train.xlsx")
export_excel(tick, "dataset")
