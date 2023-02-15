import numpy as np

class ReadAccess:
    def __init__(self, buffer, access_pattern, computation):
        # self.buffer = Buffer(defining_iterators=computation.get_parent_iterators_list(), buffer_type='a_input') # temporarily just create a new input buffer
        # self.access_pattern = computation.get_parent_iterators_list() # temporarily
        self.buffer = buffer
        self.buffer.read_list.append(self)
        self.access_pattern = access_pattern
        self.computation = computation
        self.buffer.update_wrapping_input()
        pass

    def write(self):
        text = self.buffer.wrapping_input.name + '('
        for i in range(len(self.buffer.defining_iterators)):
            if not np.any(self.access_pattern[i]):  # all the row is zeros, then the access is 0
                text += '0'
            else:  # if there is at least one non-zero value in the row
                for j in range(len(self.computation.parent_iterators_list)):
                    if self.access_pattern[i, j] == 1:  # no need for a coefficient
                        text += self.computation.parent_iterators_list[j].name + '+'
                    elif self.access_pattern[i, j] > 1:  # a coefficient is used
                        text += str(self.access_pattern[i, j]) + '*' + self.computation.parent_iterators_list[j].name + '+'
                if self.access_pattern[i, -1] > 0:  # a positive constant is used
                    text += str(self.access_pattern[i, -1])
                elif self.access_pattern[i, -1] < 0:  # a negative constant is used
                    text = text[:-1] + str(self.access_pattern[i, -1])  # remove the last + and put a minus
                else:
                    text = text[:-1]  # remove the last '+'
            text += ','
        text = text[:-1] + ')'  # remove last comma and add a parenthesis

        return text

    def write_buffer_access(self):  # This is not actually used for writing the program, just to get how the real mem access looks like
        text = self.buffer.name + '('
        for i in range(len(self.buffer.defining_iterators)):
            if not np.any(self.access_pattern[i]):  # all the row is zeros, then the access is 0
                text += '0'
            else:  # if there is at least one non-zero value in the row
                for j in range(len(self.computation.parent_iterators_list)):
                    if self.access_pattern[i, j] == 1:  # no need for a coefficient
                        text += self.computation.parent_iterators_list[j].name + '+'
                    elif self.access_pattern[i, j] > 1:  # a coefficient is used
                        text += str(self.access_pattern[i, j]) + '*' + self.computation.parent_iterators_list[
                            j].name + '+'
                if self.access_pattern[i, -1] > 0:  # a positive constant is used
                    text += str(self.access_pattern[i, -1])
                elif self.access_pattern[i, -1] < 0:  # a negative constant is used
                    text = text[:-1] + str(self.access_pattern[i, -1])  # remove the last + and put a minus
                else:
                    text = text[:-1]  # remove the last '+'
            text += ','
        text = text[:-1] + ')'  # remove last comma and add a parenthesis

        return text

    def access_pattern_is_simple(self):  # checks if all elements of the buffer are read and in order, i.e. nb_col = nb_row +1 and last_col =0 and the rest is a identity matrix
        if self.access_pattern.shape[0] + 1 != self.access_pattern.shape[1]:
            return False
        if np.any(self.access_pattern[:, -1]):
            return False
        x = self.access_pattern[:, :-1]
        if not (x == np.eye(x.shape[0])).all():
            return False
        return True
