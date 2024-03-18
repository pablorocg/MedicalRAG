import pandas as pd
import os
import xml.etree.ElementTree as ET

def dictFromXML(filename):
    QAdata = {'question': [], 'question_id': [], 'question_type': [], 'answer': [], 'focus': [], 'id': [], 'source': [], 'url': [], 'cui': [], 'semanticType': [], 'semanticGroup': []}
    # annotations = {'focus' : '', 'id' : '', 'source' : '', 'url' : '', 'cui' : '', 'semanticType': '', 'semanticGroup': ''}

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
    
    #
    # Extract annotations once, to be applied to each QA pair
    annotations = {}
    try:
        annotations['focus'] = root.find('Focus').text if root.find('Focus') is not None else ''
        annotations['id'] = root.get("id", '')
        annotations['source'] = root.get('source', '')
        annotations['url'] = root.get('url', '')
        cui_element = root.find('.//CUI')
        annotations['cui'] = cui_element.text if cui_element is not None else ''
        semantic_type_element = root.find('.//SemanticType')
        annotations['semanticType'] = semantic_type_element.text if semantic_type_element is not None else ''
        semantic_group_element = root.find('.//SemanticGroup')
        annotations['semanticGroup'] = semantic_group_element.text if semantic_group_element is not None else ''
    except Exception as e:
        print(f'Annotation error: {e}')
    #

    try:
        qaPairs = root.find('QAPairs').findall('QAPair')
    except Exception as e:
        print(f"QA pair error in {filename}: {e}")
        return QAdata

    for i in qaPairs:
        QAdata['question'].append(i.find('Question').text)
        QAdata['question_id'].append(i.find('Question').get('qid'))
        QAdata['question_type'].append(i.find('Question').get('qtype'))
        QAdata['answer'].append(i.find('Answer').text.replace('\n', ''))  # Remove newline characters
        # For each QA pair, append the same annotations
        for key in annotations.keys():
            QAdata[key].append(annotations[key])
        

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
    output_directory = './dataset/processed_data_2'

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
