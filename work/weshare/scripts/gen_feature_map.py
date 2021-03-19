#!/usr/bin/env python
# coding=utf-8

import sys
import datetime


def get_format_time():
    t = datetime.datetime.now()
    t = t.strftime('[%Y-%m-%d %H:%M:%S]')
    return t


def generate_feature_list_v1(standard_dict, *args):
    feature_map = {}
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split('\t')
        if len(tokens) < 11:
            continue
        fhash = tokens[0]
        ftype = tokens[4]
        fidx  = tokens[5]
        fname = tokens[8]
        fdim  = tokens[10]
        try:
            flen = tokens[11]
        except Exception as e:
            flen = 1
        if ftype == 'int':
            ftype = 'Int'
        elif ftype == 'sparse':
            ftype = 'Sparse'
        elif ftype == 'float' or ftype == 'double':
            ftype = 'Double'
        elif ftype == 'sequence':
            ftype = 'Sequence'
        else:
            print('%s [WARNING] Illegal type=%s found' % (get_format_time(), ftype), file=sys.stderr)
            continue
        if standard_dict is not None and len(standard_dict) > 0:
            if fname not in standard_dict:
                print('%s [WARNING] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
            if standard_dict[fname] != ftype:
                print('%s [WARNING] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
        feature_map.setdefault(fname, set())
        key = '%s,%s,%s' % (ftype, fdim, flen)
        feature_map[fname].add(key)
    for fname in feature_map:
        infos = feature_map[fname]
        if len(infos) > 1:
            print('%s [WARNING] Conflict feature found, key=%s, infos=%s' % (
                get_format_time(), fname, ' '.join(list(infos))), file=sys.stderr)
        for item in infos:
            print('%s,%s,%s' % (fname, item, fname), file=sys.stdout)


def generate_feature_list_v2(standard_dict, emb_name):
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split('\t')
        if len(tokens) < 4:
            continue
        fname = tokens[0]
        fidx  = tokens[1]
        fdim  = tokens[2]
        ftype = tokens[3]
        try:
            flen = tokens[4]
        except Exception as e:
            flen = 1
        if ftype == 'int':
            ftype = 'Int'
        elif ftype == 'sparse':
            ftype = 'Sparse'
        elif ftype == 'float' or ftype == 'double':
            ftype = 'Double'
        elif ftype == 'sequence':
            ftype = 'Sequence'
        else:
            print('%s [WARNING] Illegal type=%s found' % (get_format_time(), ftype), file=sys.stderr)
            continue
        if standard_dict is not None and len(standard_dict) > 0:
            if fname not in standard_dict:
                print('%s [WARNING] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
            if standard_dict[fname] != ftype:
                print('%s [WARNING] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
        print('%s,%s,%s,%s,%s' % (fname, ftype, fdim, flen, emb_name.get(fname, fname)), file=sys.stdout)


def generate_feature_index_v1(standard_dict, *args):
    val_cnt = 0
    conflict_cnt = 0
    feature_hash = {}
    feature_index_data = []
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split('\t')
        if len(tokens) < 11:
            continue
        fhash = tokens[0]
        feature_hash.setdefault(fhash, 0)
        feature_hash[fhash] += 1
        feature_index_data.append(tokens)
    for item in feature_index_data:
        fhash = item[0]
        ftype = item[4]
        fidx  = item[5]
        fname = item[8]
        fdim  = item[10]
        try:
            flen = item[11]
        except Exception as e:
            flen = 1
        strout = '\t'.join(item)
        if fhash not in feature_hash:
            continue
        if fname in black_features:
            continue
        if feature_hash[fhash] > 1:
            print('%s [WARNING] Conflict hash found, hash=%s' % (get_format_time(), fhash), file=sys.stderr)
            conflict_cnt += 1
            continue
        if int(fidx) >= int(fdim):
            print('%s [WARNING] index(%s) >= dim(%s)' % (get_format_time(), fidx, fdim), file=sys.stderr)
            continue
        if ftype == 'int':
            ftype = 'Int'
        elif ftype == 'sparse':
            ftype = 'Sparse'
        elif ftype == 'float' or ftype == 'double':
            ftype = 'Double'
        elif ftype == 'sequence':
            ftype = 'Sequence'
        else:
            print('%s [NOTICE] Illegal type=%s found, item=[%s]' % (get_format_time(), ftype, item_str), file=sys.stderr)
            continue
        if standard_dict is not None and len(standard_dict) > 0:
            if fname not in standard_dict:
                print('%s [NOTICE] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
            if standard_dict[fname] != ftype:
                print('%s [NOTICE] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
            print(strout, file=sys.stdout)
        val_cnt += 1
    print('%s [NOTICE] Conflict ratio = %.2f%%' % (
        get_format_time(), conflict_cnt * 1.0 / len(feature_index_data) * 100), file=sys.stderr)
    print('%s [NOTICE] %d/%d lines written' % (
        get_format_time(), val_cnt, len(feature_index_data)), file=sys.stderr)

def generate_feature_index_v2(standard_dict, *args):
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split('\t')
        if len(tokens) < 4:
            continue
        fname = tokens[0]
        fidx  = tokens[1]
        fdim  = tokens[2]
        ftype = tokens[3]
        try:
            flen  = tokens[4]
        except Exception as e:
            flen  = 1
        if ftype == 'int':
            ftype = 'Int'
        elif ftype == 'sparse':
            ftype = 'Sparse'
        elif ftype == 'float' or ftype == 'double':
            ftype = 'Double'
        elif ftype == 'sequence':
            ftype = 'Sequence'
        else:
            print('%s [WARNING] Illegal type=%s found' % (get_format_time(), ftype), file=sys.stderr)
            continue
        if standard_dict is not None and len(standard_dict) > 0:
            if fname not in standard_dict:
                print('%s [WARNING] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
            if standard_dict[fname] != ftype:
                print('%s [WARNING] Ignore feature=(%s,%s)' % (get_format_time(), fname, ftype), file=sys.stderr)
                continue
        if  ftype == 'Double':
            ftype = 'float'
        elif ftype == 'Sequence':
            ftype = 'sequence'
        elif ftype == 'Int':
            ftype = "int"
        else:
            continue
        print('%s\t%s\t%s\t%s\t%s' % (fname, ftype, fdim, flen, fname), file=sys.stdout)


def main():
    if len(sys.argv) != 5:
        print('%s [NOTICE] Usage: %s <input_index_file> <task_type> <black_features> <is_share_emb>' % (
            get_format_time(), sys.argv[0]
        ), file=sys.stderr)
        sys.exit(1)
    try:
        is_share_emb = True if sys.argv[4] == 'True' else False
        std_index_file = sys.argv[1]
        standard_dict = dict()
        emb_name = dict()
        for line in open(std_index_file, 'rt'):
            line = line.strip()
            tokens = line.split('\t')
            if len(tokens) < 2:
                continue
            elif len(tokens) > 2 and is_share_emb:
                emb_name[tokens[0]] = tokens[2]
            standard_dict[tokens[0]] = tokens[1]
        print('%s [NOTICE] %d standard features found' % (get_format_time(), len(standard_dict)), file=sys.stderr)
    except Exception as e:
        standard_dict = None
        print('%s [WARNING] Exception found, e=%s' % (get_format_time(), e), file=sys.stderr)
        sys.exit(1)
    try:
        black_features = set(sys.argv[3].split(','))
        print('%s [NOTICE] black_features="%s" found' % (get_format_time(), sys.argv[3]), file=sys.stderr)
    except Exception as e:
        black_features = set()
        print('%s [WARNING] Exception found, e=%s, black_features=%s' % (get_format_time(), e, sys.argv[3]), file=sys.stderr)
    func_cmd = 'generate_' + sys.argv[2] + '(standard_dict, emb_name)'
    eval(func_cmd)


if __name__ == '__main__':
    main()
    sys.exit(0)
