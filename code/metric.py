from sklearn.metrics import f1_score, accuracy_score, classification_report

def get_zero_metrics(setting):
    if setting.use_extra_label:
        ret_dict =  {
            'big_accr': 0,
            'big_f1': 0,
            'middle_accr': 0,
            'middle_f1': 0,
            'accr': 0,
            'f1': 0,
        }

        if setting.el_single_output:
            ret_dict['small_accr'] = 0
            ret_dict['small_f1'] = 0
    else:
        ret_dict = {
            'accr': 0,
            'f1': 0,
        }
    
    return ret_dict

def get_metrics(setting, y_true, y_pred_dict):
    if setting.use_extra_label:
        big_y_true = y_true[:, 0]
        middle_y_true = y_true[:, 1]
        y_true = y_true[:, 2]

        big_y_pred = y_pred_dict['big_pred']
        middle_y_pred = y_pred_dict['middle_pred']
        y_pred = y_pred_dict['pred']
        

        ret_dict =  {
            'big_accr': accuracy_score(big_y_true, big_y_pred),
            'big_f1': f1_score(big_y_true, big_y_pred, average='macro'),
            'middle_accr': accuracy_score(middle_y_true, middle_y_pred),
            'middle_f1': f1_score(middle_y_true, middle_y_pred, average='macro'),
            'accr': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='macro'),
        }

        if setting.el_single_output:
            small_y_pred = y_pred_dict['small_pred']
            ret_dict['small_accr'] = accuracy_score(y_true, small_y_pred)
            ret_dict['small_f1'] = f1_score(y_true, small_y_pred, average='macro')
    else:
        y_pred = y_pred_dict['pred']
        accr = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        ret_dict = {
            'accr': accr,
            'f1': f1,
        }
    
    return ret_dict

def show_classification_report(setting, y_true, y_pred_dict):
    if setting.use_extra_label:
        big_y_true = y_true[:, 0]
        middle_y_true = y_true[:, 1]
        y_true = y_true[:, 2]

        big_y_pred = y_pred_dict['big_pred']
        middle_y_pred = y_pred_dict['middle_pred']
        y_pred = y_pred_dict['pred']

        print('BIG')
        print(classification_report(big_y_true, big_y_pred))
        print()
        
        print('MIDDLE')
        print(classification_report(middle_y_true, middle_y_pred))
        print()

        print('NORMAL')
        print(classification_report(y_true, y_pred))
        print()
    else:
        y_pred = y_pred_dict['pred']
        print('NORMAL')
        print(classification_report(y_true, y_pred))
        print()