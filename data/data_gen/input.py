class Input:
    def __init__(self, buffer, program):
        self.buffer = buffer
        self.defining_iterators = self.buffer.defining_iterators.copy()
        if self.buffer.write_list != []:
            self.name = 'i' + self.buffer.write_list[-1].computation.name
        else:
            self.name = 'input' + self.buffer.name[-2:]
        self.program = program
        self.program.input_list.append(self)
        self.data_type = 'p_float64'  # temporarily fixed data type

    def write(self):
        input_line = '\tinput ' + self.name + '("' + self.name + '", {'
        for iterator in self.defining_iterators:
            input_line += iterator.name + ','
        input_line = input_line[:-1]  # remove last comma
        input_line += '}, ' + self.data_type + ');\n'
        return input_line

    def write_store(self):
        return '\t' + self.name + '.store_in(&' + self.buffer.name + ');\n'

