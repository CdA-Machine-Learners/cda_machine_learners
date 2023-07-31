import math, re, uuid, hashlib


def xlist( ary ):
    if ary is None:
        return []
    if isinstance(ary, str):
        return ','.split(ary)
    return list(ary)


def xtuple( tup ):
    return tuple(tup) if tup is not None else tuple()


def xstr( s, none='' ):
    return str(s) if s is not None else none


def xint( s, none=0, undefined=None ):
    try:
      if s == "undefined":
        return undefined
      return int(s) if s is not None and s != 'NaN' else none
    except ValueError:
        #Floating points and trailing letters wont fool me!!!
        m = re.search('^[-+]?[0-9]+', s)
        if m:
            return int(m.group(0))

        #can't go any further
        return none
    except TypeError:
        return none


def xfloat( s, none=0.0, undefined=None ):
    try:
        if s == "undefined":
            return undefined
        f = float(s) if s is not None and s != 'NaN' else none
        if math.isnan(f):
            return none
        return f
    except ValueError:
        #trailing letters wont fool me!!!
        m = re.search('^[-+]?[0-9]*\.?[0-9]+', s )
        if m:
            return float(m.group(0))

        #Can't go any further
        return none
    except TypeError:
        return none


def xbool( s, none=False, undefined=False ):
    #Are we string? try to figure out what that means
    if isinstance( s, str ):
        s = s.lower()
        if s == 'true':
            return True
        elif s == 'none' or s == 'null':
            return none
        elif s == 'undefined':
            return undefined
        else:
            return False

    #Special case none
    elif s is None:
        return none
    else:
        return bool(s)


def xlen( x, none=0 ):
    return len(x) if x is not None else none


def is_valid_guid(guid):
    regex = r'^\b[A-Fa-f0-9]{8}(?:-[A-Fa-f0-9]{4}){3}-[A-Fa-f0-9]{12}\b$'
    return xbool(re.match(regex, guid))


def cap( value, largest, smallest=0 ):
    if isinstance( value, int ) or isinstance( value, float ):
        value = round( value )
    return max( min( xint( value ), largest ), smallest )


def upperCaseFirst( s ):
    return ' '.join([x.capitalize() for x in s.split(' ')])


def snakeToCamel( s ):
    return ''.join( [x.capitalize() for x in xstr(s).split('_')])


def camelToSnake( s ):
    s = xstr(s)
    return (s[0] + re.sub('([A-Z])', r'_\1', s[1:])).lower()


# Cap a value between the given bounds
def cap( val, high, low=None ):
    if not low:
        low = -high
    if val > high:
        val = high
    elif val < low:
        val = low

    return val


# Generate reset code
def hash_code():
    # Create the reset code
    m = hashlib.sha256()
    m.update(b"A super important message that must happen")
    m.update(bytes(str(uuid.uuid4()), 'utf-8'))

    # Save the pwd reset code
    return m.hexdigest()


def get_ip_address(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip