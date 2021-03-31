from freq_2500 import freq_list
from less_freq_1000 import less_freq_list
from letter import chars_list, spec_list


def get_all_keys():
    return freq_list + less_freq_list + chars_list + spec_list


N_CLASS = len(get_all_keys())

if __name__ == '__main__':
    print(N_CLASS)
