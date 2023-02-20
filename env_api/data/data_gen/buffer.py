from .input import Input
import random

class Buffer:
    def __init__(self, defining_iterators, program):
        # optimally, only buffers that are read before being written on need an input wrapper (accesses as input) but to avoid complications, we create an input for each buffer with at least one read access
        self.name = 'buf' + str(len(program.buffer_list)).zfill(2)  # TODO find a naming convention for buffers
        self.type = ''  # either a_output, a_input, a_temporary, to define when writing
        self.read_list = []  # a list of ReadAccess that are used to read from the buffer
        self.write_list = []  # a list of WriteAccess that are used to write on the buffer
        self.defining_iterators = defining_iterators  # the iterator who's sizes are used to define the buffer
        self.sizes = [i.upper_bound - i.lower_bound for i in defining_iterators]
        self.wrapping_input = None  # the input wrapper that will be used to read from the buffer. will be defined a write
        self.program = program
        self.program.buffer_list.append(self)
        self.data_type = 'p_float64'  # temporarily fixed

    # def fit_size_to_accesses(self):  # will be called when a stencil access (optimaly, but it can be called for every access) is used on the buf, this will correct the extent
    #     # must also create new defining iterators for the wrapping input, and add them to the programs iterators list
    #     for read_access in self.read_list:
    #         if np.any(read_access.access_pattern[:, -1]):
    #             for i in range(len(read_access.access_pattern[:, -1])):
    #                 if self.sizes[i] < (self.wrapping_input.defining_iterators[i].upper_bound - self.wrapping_input.defining_iterators[i].lower_bound) + read_access.access_pattern[i, -1]:
    #                     self.sizes[i] = (self.defining_iterators[i].upper_bound - self.defining_iterators[i].lower_bound) + read_access.access_pattern[i, -1]
    #                     # print(self.wrapping_input.defining_iterators[i].name)
    #                     # if '_p' in self.wrapping_input.defining_iterators[i].name:  # if the iterator has already been extended
    #                     #     self.program.iterators_list.remove(self.wrapping_input.defining_iterators[i])
    #                     self.wrapping_input.defining_iterators[i] = \
    #                         Loop.input_iterator(lower_bound=self.defining_iterators[i].lower_bound, upper_bound=self.defining_iterators[i].upper_bound + read_access.access_pattern[i, -1], name=self.defining_iterators[i].name+'_p'+str(read_access.access_pattern[i, -1]), program=self.program)
    #     pass

    def update_wrapping_input(self):  # will be called for every read access created
        if self.wrapping_input is None:
            self.wrapping_input = Input(self, self.program)
        # self.fit_size_to_accesses()

    def write(self):
        if self.write_list != []:  # updating the buffer type
            self.type = 'a_output'
        else:
            self.type = 'a_input'
        buffer_declaration = '\tbuffer ' + self.name + '("' + self.name + '", {' + \
                             ','.join([str(size) for size in self.sizes]) + '}, ' + self.data_type + \
                             ', ' + self.type + ');\n'
        return buffer_declaration

    def write_wrapper(self):
        dtype = ''
        if self.data_type == 'p_float64':
            dtype = 'double'
        else:
            raise Exception('unrecognized data type')

        text = ''
        if self.type == 'a_input' or self.read_list != []:  # if input or read
            text += '\t' + dtype + ' *c_' + self.name + ' = (' + dtype + '*)malloc(' + '*'.join([str(size) for size in self.sizes[::-1]]) + '* sizeof(' + dtype + '));\n'
            text += '\tparallel_init_buffer(c_' + self.name + ', ' + '*'.join([str(size) for size in self.sizes[::-1]]) + ', (' + dtype + ')' + str(random.randint(1, 100)) + ');\n'
            text += '\tHalide::Buffer<' + dtype + '> ' + self.name + '(c_' + self.name + ', ' + ','.join([str(size) for size in self.sizes[::-1]]) + ');\n'
        elif self.type == 'a_output':
            text += '\tHalide::Buffer<' + dtype + '> ' + self.name + '(' + ','.join([str(size) for size in self.sizes[::-1]]) + ');\n'

        return text

