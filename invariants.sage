from snappy import OrientableCuspedCensus

prec = 500             # bit precision for computations
max_cusps = 10         # maximum number of cusps for which to report invariants
max_degree = 15        # maximum degree for (invariant) trace field
zeta_values = [2, 3, 4, 5, 6, 7]   # numbers at which to evaluate zeta function
zeta_coeffs = 10       # number of coefficients of the zeta function to include
# primes at which to report the discriminant's valuation
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


# Return the cardinality of the torsion part of a finitely generated
# abelian group.
def torsion_card(A):
    return prod(x for x in A.elementary_divisors() if x != 0)


# Check if a polynomial has only integer coefficients. For monic
# polynomials, this means being an integral polynomial.
def is_integral_polynomial(f):
    return all(c.is_integer() for c in f.coefficients())


# def zeta_residue(min_pol):
#     return pari('Col(lfunrootres(lfuncreate(' +str(min_pol) + '))[1][1][2])[1]')


# Compute the residue at one numerically.
def residue(zeta):
    eps = 1e-10
    return zeta(1 + eps) * eps


# Truncate a list to the desired length, then pad with zeros if
# needed.
def zero_pad(l, length):
    if len(l) >= length:
        return l[:length]
    else:
        return l + [0] * (length - len(l))


# loop all census manifolds and report their invariants
for i, M in enumerate(OrientableCuspedCensus):
    print(M,
          M.volume(),
          M.num_cusps(),
          M.chern_simons(),
          *[RR(x) for x in
            zero_pad(M.cusp_areas(method='maximal',
                                  verified=True,
                                  policy='unbiased',
                                  bits_prec=prec),
                     max_cusps)],
          # M.cusp_info(0)['modulus'],
          *[0 if i >= M.num_cusps() else M.cusp_info(i)['shape'].real()
            for i in range(max_cusps)],
          *[0 if i >= M.num_cusps() else M.cusp_info(i)['shape'].imag()
            for i in range(max_cusps)],
          M.homology().betti_number(),
          torsion_card(M.homology()),
          1 if M.is_two_bridge() else 0,
          # M.length_spectrum(), sep='\n'
          )

    # compute numerical approximations for generators of the
    # (invariant) trace field and find defining polynomials for the
    # (invariant) trace field using lattice techniques
    gens_L = M.trace_field_gens()
    L = gens_L.find_field(prec=prec, degree=max_degree, optimize=True)
    L = L[0] if L else None
    gens_K = M.invariant_trace_field_gens()
    K = gens_K.find_field(prec=prec, degree=max_degree, optimize=True)
    K = K[0] if K else None
    # report arithmetic invariants of both fields
    for desc, gens, k in [('inv_tr_fld', gens_K, K), ('tr_fld', gens_L, L)]:
        try:
            minpols = [gens[i].min_polynomial(prec=prec, degree=max_degree)
                       for i in range(gens.n)]
            print(desc,
                  k.degree(),
                  *k.signature(),
                  k.discriminant(),
                  *[k.discriminant().valuation(p) for p in primes],
                  k.number_of_roots_of_unity(),
                  k.regulator(proof=False),
                  k.class_number(proof=False),
                  *[k.zeta_function()(i) for i in zeta_values],
                  *k.zeta_coefficients(zeta_coeffs),
                  # zeta_residue(k.polynomial()),
                  # residue(k.zeta_function()),
                  1 if all(minpols) else 0,
                  1 if all(minpols) and
                  all(is_integral_polynomial(f) for f in minpols) else 0)
        except Exception as e:
            print(desc, "missing", e)
            continue
