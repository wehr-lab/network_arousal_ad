def strcmp(str1, str2):
    if str2 is not str or type(str2) in [list, tuple]:
        return str1 in str2 
    else:
        return str1 == str2