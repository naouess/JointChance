#=
    TODO
=#
function chi_cdf_gamma(x, c, gm)

    if x <= 0.0
        cdf = 0.0
    else
        x2 = 0.5 * x * x
        p2 = 0.5 * c
        
        cdf = my_gamma_inc(p2, x2, gm)
    end
    
    return cdf
end

#=
    Implementation of the gamma_inc function 
    TODO
=#
function my_gamma_inc(p, x, gm)

    exp_arg_min = -88.0
    overflow = 1.0E+37
    plimit = 10000.0
    tol = 1.0E-07
    xbig = 1.0E+08

    value = 0.0

    if p <= 0.0
        println("\n")
        println("GAMMA_INC - Fatal error!")
        println("  Parameter P <= 0.")
        error("GAMMA_INC - Fatal error!")
    end

    if x <= 0.0
        value = 0.0
        return value
    end

    # Use a normal approximation if PLIMIT < P.
    if plimit < p
        pn1 = 3.0 * sqrt(p) * ((x / p)^(1.0 / 3.0) + 1.0 / (9.0 * p) - 1.0)
        cdf = cdf(Normal(), pn1)
        value = cdf
        return value
    end

    # Is X extremely large compared to P?
    if xbig < x
        value = 1.0
        return value
    end

    # Use Pearson's series expansion.
    # (P is not large enough to force overflow in the log of Gamma.)
    if x <= 1.0 || x < p
        arg = p * log(x) - x - gm[2]
        c = 1.0
        value = 1.0
        a = p

        while true
            a += 1.0
            c *= x / a
            value += c
            if c <= tol
                break
            end
        end

        arg += log(value)

        if exp_arg_min <= arg
            value = exp(arg)
        else
            value = 0.0
        end
    else
        # Use a continued fraction expansion.
        arg = p * log(x) - x - gm[1]
        a = 1.0 - p
        b = a + x + 1.0
        c = 0.0
        pn1 = 1.0
        pn2 = x
        pn3 = x + 1.0
        pn4 = x * b
        value = pn3 / pn4

        while true
            a += 1.0
            b += 2.0
            c += 1.0
            pn5 = b * pn3 - a * c * pn1
            pn6 = b * pn4 - a * c * pn2

            if abs(pn6) > 0.0
                rn = pn5 / pn6

                if abs(value - rn) <= min(tol, tol * rn)
                    arg += log(value)
                    if exp_arg_min <= arg
                        value = 1.0 - exp(arg)
                    else
                        value = 1.0
                    end

                    return value
                end

                value = rn
            end

            pn1 = pn3
            pn2 = pn4
            pn3 = pn5
            pn4 = pn6

            # Rescale terms in continued fraction if terms are large.
            if abs(pn5) >= overflow
                pn1 /= overflow
                pn2 /= overflow
                pn3 /= overflow
                pn4 /= overflow
            end
        end
    end
end  

#=
    TODO
=#
function chi_pdf_gamma(x, c, gm)
   
    pdf = 0.0
    
    if (0 < x < 100)
        y = 0.5 * x * x
        p = 0.5 * c
        
        pdf = exp(-y) * x^(c - 1.0) / (2.0^(p - 1.0) * gm[3])
    end
    
    return pdf
end
