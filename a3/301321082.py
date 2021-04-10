from efficient_apriori import apriori


def get_data_list(path):   
    data = []
    with open(path, 'r') as filestream:
        for line in filestream:
            # remove \n from the string line and then split for each space
            items = line.rstrip().split(' ')

            temp = []
            for item in items:
                if item == '-1' or item == '-2':
                    # skip item -1 and -2 item values
                    continue    

                temp.append(item)
            
            data.append(temp)

    return data



if __name__ == "__main__":
    data = get_data_list('BMS2.txt')
    
    itemsets, rules = apriori(data, min_support=0.005, min_confidence=0.7)
    print('type: ', type(rules))
    print('len: ', len(rules))
