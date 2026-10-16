import cmath


def clean_complex(val, tol=1e-10):
    """Removes negligible real/imaginary parts due to floating-point rounding."""
    real = val.real if abs(val.real) > tol else 0.0
    imag = val.imag if abs(val.imag) > tol else 0.0
    if imag == 0:
        return round(real, 6)
    return complex(round(real, 6), round(imag, 6))


def solve_cubic(a, b, c, d):
    """Solves ax^3 + bx^2 + cx + d = 0 using Cardano's Method."""
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be 0 for a cubic equation.")

    b_a, c_a, d_a = b / a, c / a, d / a

    p = c_a - (b_a**2) / 3
    q = (2 * (b_a**3)) / 27 - (b_a * c_a) / 3 + d_a

    discriminant = (q / 2) ** 2 + (p / 3) ** 3
    sqrt_disc = cmath.sqrt(discriminant)

    u = (
        cmath.exp(cmath.log(-q / 2 + sqrt_disc) / 3)
        if (-q / 2 + sqrt_disc) != 0
        else 0
    )
    v = (
        cmath.exp(cmath.log(-q / 2 - sqrt_disc) / 3)
        if (-q / 2 - sqrt_disc) != 0
        else 0
    )

    w1 = complex(-0.5, cmath.sqrt(3) / 2)
    w2 = complex(-0.5, -cmath.sqrt(3) / 2)

    t1 = u + v
    t2 = u * w1 + v * w2
    t3 = u * w2 + v * w1

    shift = b_a / 3
    roots = (t1 - shift, t2 - shift, t3 - shift)

    return tuple(clean_complex(root) for root in roots)


def solve_quartic(a, b, c, d, e):
    """Solves ax^4 + bx^3 + cx^2 + dx + e = 0 using Ferrari's Method."""
    if a == 0:
        # If 'a' is 0, fall back to cubic solver
        return solve_cubic(b, c, d, e)

    # 1. Normalize coefficients: x^4 + A*x^3 + B*x^2 + C*x + D = 0
    A, B, C, D = b / a, c / a, d / a, e / a

    # 2. Depress the quartic: substitute x = y - A/4
    # Form: y^4 + p*y^2 + q*y + r = 0
    p = B - (3 * A**2) / 8
    q = C - (A * B) / 2 + (A**3) / 8
    r = D - (A * C) / 4 + (A**2 * B) / 16 - (3 * A**4) / 256

    # Special case: if q == 0, equation is biquadratic (y^4 + p*y^2 + r = 0)
    if abs(q) < 1e-11:
        # Solve z^2 + p*z + r = 0 where z = y^2
        disc = cmath.sqrt(p**2 - 4 * r)
        z1 = (-p + disc) / 2
        z2 = (-p - disc) / 2
        
        y_roots = [
            cmath.sqrt(z1), -cmath.sqrt(z1),
            cmath.sqrt(z2), -cmath.sqrt(z2)
        ]
    else:
        # 3. Resolvent Cubic in m: 8*m^3 + 8*p*m^2 + (2*p^2 - 8*r)*m - q^2 = 0
        cubic_roots = solve_cubic(8, 8 * p, 2 * p**2 - 8 * r, -(q**2))
        
        # Pick any root m (preferably real/non-zero)
        m = cubic_roots[0]
        for root in cubic_roots:
            if abs(root.imag) < 1e-6 and abs(root) > 1e-6:
                m = root.real
                break

        # 4. Calculate factors for quadratic split
        sqrt_2m = cmath.sqrt(2 * m)
        
        # Solve two quadratics: y^2 + sqrt(2m)*y + (p/2 + m - q/(2*sqrt(2m))) = 0
        # and                  y^2 - sqrt(2m)*y + (p/2 + m + q/(2*sqrt(2m))) = 0
        k1 = p / 2 + m - q / (2 * sqrt_2m)
        k2 = p / 2 + m + q / (2 * sqrt_2m)

        y1 = (-sqrt_2m + cmath.sqrt(2 * m - 4 * k1)) / 2
        y2 = (-sqrt_2m - cmath.sqrt(2 * m - 4 * k1)) / 2
        y3 = (sqrt_2m + cmath.sqrt(2 * m - 4 * k2)) / 2
        y4 = (sqrt_2m - cmath.sqrt(2 * m - 4 * k2)) / 2

        y_roots = [y1, y2, y3, y4]

    # 5. Shift back to x = y - A/4
    shift = A / 4
    x_roots = tuple(clean_complex(y - shift) for y in y_roots)
    return x_roots


def main():
    print("=== Polynomial Equation Solver ===")
    
    # Example: Solving x^4 - 10x^3 + 35x^2 - 50x + 24 = 0
    # Roots for this equation are x = 1, 2, 3, 4
    a, b, c, d, e = 1,0,0, 0, -1
    
    print(f"Solving: {a}x⁴ + ({b})x³ + ({c})x² + ({d})x + ({e}) = 0\n")
    
    roots = solve_quartic(a, b, c, d, e)
    
    print("The 4 roots are:")
    for idx, root in enumerate(roots, start=1):
        print(f"  x{idx} = {root}")


if __name__ == "__main__":
    main()