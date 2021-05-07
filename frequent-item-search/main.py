from efficient_apriori import apriori

class ClosedMaxItemset:
    def __init__(self):
        self.is_closed = True
        self.is_max = True
        self.closed_item_sets = dict()
        self.max_item_sets = dict()


    def get_data(self, path):
        data = list()
        with open(path, 'r') as filestream:
            for line in filestream:
                items = line.rstrip().split(' ')

                temp = list()
                for item in items:
                    if item == '-1' or item == '-2':
                        continue
                    temp.append(item)
                
                data.append(temp)

        return data


    def get_frequent_itemset(self, data):
        itemsets, rules = apriori(data, min_support=0.005, min_confidence=0.7)
        return itemsets


    def is_valid_itemset(self, original_set, temp_set, original_count, temp_count):
        # if original item set is a subset of target set then no maximal
        if original_set.issubset(temp_set):
            self.is_max = False

            # for a set to be closed, it needs to have more elements than its subset
            if original_count == temp_count:
                self.is_closed = False                        

            # item set is invalid (neither closed or max)
            if not self.is_max and not self.is_closed:
                return False

        return True


    def get_closed_and_max_itemsets(self, itemsets):
        # transform the nested dictionary, itemsets, to a single dictionary
        d = dict()
        index = 1
        while index <= len(itemsets):
            for k,v in itemsets.get(index).items():
                d[k] = v
            index += 1       

        for items, count in d.items():
            original_set = set(items)
            self.is_closed = True
            self.is_max = True
            
            # compare item sets against each other
            for copy_items, temp_count, in d.items():
                # no need to compare a set against itself
                if items == copy_items:
                    continue

                temp_set = set(copy_items)
                
                # a valid itemset qualifies as either a max or closed itemset
                if not self.is_valid_itemset(original_set, temp_set, count, temp_count):
                    break
                else:
                    pass
            
            # identify itemsets as closed or max
            if self.is_max:
                key = len(original_set)
                self.max_item_sets.setdefault(key, {}).update({items : count})

            if self.is_closed:
                key = len(original_set)
                self.closed_item_sets.setdefault(key, {}).update({items : count})
        
        return self.closed_item_sets, self.max_item_sets


    def get_itemset_size(self, itemsets):
        length = 0
        for num, items in itemsets.items():
            length += len(items)

        return length


if __name__ == '__main__':
    path = 'BMS2.txt'
    cm = ClosedMaxItemset()
    data = cm.get_data(path)
    itemsets = cm.get_frequent_itemset(data)
    closed_item_sets, max_item_sets = cm.get_closed_and_max_itemsets(itemsets)

    print('frequent item sets length: ', cm.get_itemset_size(itemsets))
    print('closed item sets length: ', cm.get_itemset_size(closed_item_sets))
    print(closed_item_sets)
    print('\n\nmax item sets length: ', cm.get_itemset_size(max_item_sets))
    print(max_item_sets)
