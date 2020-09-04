def g_factor(J, L, S):
    j = J*(J+1)
    l = L*(L+1)
    s = S*(S+1)
    try:
        g = 3/2 + 1/2 * ((s-l)/j)
    except:
        g = 0.
    return g

def effective_Bohr_magneton(J, L, S):
    j = J*(J+1)
    g = g_factor(J, L, S)
    p = g * j ** 0.5
    return p

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("-J", dest="J", default=2.5, type="float",
                      help="Total angular momentum, REQUIRED",
                      metavar="J")
    parser.add_option("-L", dest="L", default=3, type="float",
                      help="Orbital angular momentum, REQUIRED",
                      metavar="L")
    parser.add_option("-S", dest="S", default=0.5, type="float",
                      help="Spin angular momentum, REQUIRED",
                      metavar="S")
    (options, args) = parser.parse_args()

    p = effective_Bohr_magneton(options.J, options.L, options.S)
    print(p)
