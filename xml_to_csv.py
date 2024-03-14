# import pandas as pd
# import os
# import xml.etree.ElementTree as ET

# def dictFromXML(filename):
    
#     QAdata = {'question': [], 'question_id' : [], 'question_type' : [], 'answer' : []}

#     try:
#         tree = ET.parse(filename)
#     except ET.ParseError as e:
#         print(f"XML parsing error: {e}")
#     except FileNotFoundError:
#         print("File not found")
#     except Exception as e:
#         print(f"An error occurred: {e}")

   

#     root = tree.getroot() 

#     #handling incorrect formatting by printing an error message at that formatting issue then moving on
#     try:
#         qaPairs = root.find('QAPairs').findall('QAPair')
#     except:
#         print(f"QA pair error at:{filename}")
#         return

#     for i in qaPairs:
#         QAdata['question'].append(i.find('Question').text) 
#         QAdata['question_id'].append(i.find('Question').get('qid'))
#         QAdata['question_type'].append(i.find('Question').get('qtype'))
#         QAdata['answer'].append(i.find('Answer').text)

    

#     #returning the data as a dict
#     return QAdata


# #defining a function that converts a whole folder of xmls to dataframes and combines them
# def XMLsToDataFrame(directory):
#     df = pd.DataFrame()
#     dataframes = []

#     for filename in os.listdir(directory):
#         if filename.endswith('.xml'):
#             with open(os.path.join(directory, filename)) as f:
#                 data = dictFromXML(f)
#                 dataframes.append(pd.DataFrame(data))

#     try: 
#         df = pd.concat(dataframes)
#     except:
#         print("could not concat")
#     return df


# if __name__ == "__main__":

#     directory = './dataset/MedQuAD-master'

#     print(os.listdir(directory))
#     for dirname in os.listdir(directory):
#         print(dirname)
#         try:
#             print(f'{directory}/{dirname}')
#             df = XMLsToDataFrame(f'{directory}/{dirname}')
#             df.to_csv(f'.dataset/processed_data/{dirname}.csv')
#         except:
#             print('could not execute function')

import pandas as pd
import os
import xml.etree.ElementTree as ET

def dictFromXML(filename):
    QAdata = {'question': [], 'question_id': [], 'question_type': [], 'answer': [], 'url': []}
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return QAdata  # Return empty data structure
    except FileNotFoundError:
        print("File not found")
        return QAdata
    except Exception as e:
        print(f"An error occurred: {e}")
        return QAdata

    try:
        qaPairs = root.find('QAPairs').findall('QAPair')
    except Exception as e:
        print(f"QA pair error in {filename}: {e}")
        return QAdata

    for i in qaPairs:
        QAdata['question'].append(i.find('Question').text)
        QAdata['question_id'].append(i.find('Question').get('qid'))
        QAdata['question_type'].append(i.find('Question').get('qtype'))
        QAdata['answer'].append(i.find('Answer').text)
        

    return QAdata

def XMLsToDataFrame(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            fullPath = os.path.join(directory, filename)
            data = dictFromXML(fullPath)
            if data['question']:  # Check if data is not empty
                dataframes.append(pd.DataFrame(data))

    try:
        df = pd.concat(dataframes, ignore_index=True)
    except ValueError as e:
        print(f"Could not concatenate dataframes: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame if concatenation fails
    return df

if __name__ == "__main__":
    directory = './dataset/MedQuAD-master'
    output_directory = './dataset/processed_data'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for dirname in os.listdir(directory):
        dirPath = os.path.join(directory, dirname)
        if os.path.isdir(dirPath):
            print(f'Processing directory: {dirPath}')
            try:
                df = XMLsToDataFrame(dirPath)
                if not df.empty:
                    output_path = os.path.join(output_directory, f'{dirname}.csv')
                    df.to_csv(output_path, index=False)
                    print(f'Data saved to {output_path}')
                else:
                    print(f'No data to save for directory: {dirPath}')
            except Exception as e:
                print(f'Could not execute function for directory {dirPath}: {e}')
