from compiler.program import Program, CommonPreprocessedInput
from utils import *
from setup import *
from typing import Optional
from dataclasses import dataclass
from transcript import Transcript, Message1, Message2, Message3, Message4, Message5
from poly import Polynomial, Basis

@dataclass
class Proof:
    msg_1: Message1
    msg_2: Message2
    msg_3: Message3
    msg_4: Message4
    msg_5: Message5

    def flatten(self):
        proof = {}
        proof["a_1"] = self.msg_1.a_1
        proof["b_1"] = self.msg_1.b_1
        proof["c_1"] = self.msg_1.c_1
        proof["z_1"] = self.msg_2.z_1
        proof["t_lo_1"] = self.msg_3.t_lo_1
        proof["t_mid_1"] = self.msg_3.t_mid_1
        proof["t_hi_1"] = self.msg_3.t_hi_1
        proof["a_eval"] = self.msg_4.a_eval
        proof["b_eval"] = self.msg_4.b_eval
        proof["c_eval"] = self.msg_4.c_eval
        proof["s1_eval"] = self.msg_4.s1_eval
        proof["s2_eval"] = self.msg_4.s2_eval
        proof["z_shifted_eval"] = self.msg_4.z_shifted_eval
        proof["W_z_1"] = self.msg_5.W_z_1
        proof["W_zw_1"] = self.msg_5.W_zw_1
        return proof


@dataclass
class Prover:
    group_order: int
    setup: Setup
    program: Program
    pk: CommonPreprocessedInput

    def __init__(self, setup: Setup, program: Program):
        self.group_order = program.group_order
        self.setup = setup
        self.program = program
        self.pk = program.common_preprocessed_input()

    def prove(self, witness: dict[Optional[str], int]) -> Proof:
        # Initialise Fiat-Shamir transcript
        transcript = Transcript(b"plonk")

        # Collect fixed and public information
        # FIXME: Hash pk and PI into transcript
        public_vars = self.program.get_public_assignments()
        PI = Polynomial(
            [Scalar(-witness[v]) for v in public_vars]
            + [Scalar(0) for _ in range(self.group_order - len(public_vars))],
            Basis.LAGRANGE,
        )
        self.PI = PI

        # Round 1
        msg_1 = self.round_1(witness)
        self.beta, self.gamma = transcript.round_1(msg_1)

        # Round 2
        msg_2 = self.round_2()
        self.alpha, self.fft_cofactor = transcript.round_2(msg_2)

        # Round 3
        msg_3 = self.round_3()
        self.zeta = transcript.round_3(msg_3)

        # Round 4
        msg_4 = self.round_4()
        self.v = transcript.round_4(msg_4)

        # Round 5
        msg_5 = self.round_5()

        return Proof(msg_1, msg_2, msg_3, msg_4, msg_5)

    def round_1(
        self,
        witness: dict[Optional[str], int],
    ) -> Message1:
        program = self.program
        setup = self.setup
        group_order = self.group_order

        if None not in witness:
            witness[None] = 0

        # Compute wire assignments for A, B, C, corresponding:
        # - A_values: witness[program.wires()[i].L]
        # - B_values: witness[program.wires()[i].R]
        # - C_values: witness[program.wires()[i].O]
        self.values = [[Scalar(witness[name]) for name in wire.as_list()] for wire in program.wires()]
        poly_values = zip(*self.values)

        blinding_const = 5
        polys = [Polynomial(value + (Scalar(0),)*blinding_const, Basis.LAGRANGE) for value in poly_values]
        
        self.A = polys[0]
        self.B = polys[1]
        self.C = polys[2]

        commitments = [setup.commit(poly) for poly in polys]

        # Sanity check that witness fulfils gate constraints
        assert (
            self.A * self.pk.QL
            + self.B * self.pk.QR
            + self.A * self.B * self.pk.QM
            + self.C * self.pk.QO
            + self.PI
            + self.pk.QC
            == Polynomial([Scalar(0)] * group_order, Basis.LAGRANGE)
        )

        # Return a_1, b_1, c_1
        return Message1(*commitments)

    def round_2(self) -> Message2:
        group_order = self.group_order
        setup = self.setup

        # Using A, B, C, values, and pk.S1, pk.S2, pk.S3, compute
        # Z_values for permutation grand product polynomial Z
        #
        # Note the convenience function:
        #       self.rlc(val1, val2) = val_1 + self.beta * val_2 + gamma


        Z_values = []
        roots_of_unity = Scalar.roots_of_unity(self.pk.group_order)

        for i in range(self.pk.group_order):
            prod = Scalar(1)
            for j in range(i):
                m = roots_of_unity[j]
                prod *= self.rlc(self.A.values[j], m)
                prod *= self.rlc(self.B.values[j], 2*m)
                prod *= self.rlc(self.C.values[j], 3*m)
                prod /= self.rlc(self.A.values[j], self.pk.S1.values[j])
                prod /= self.rlc(self.B.values[j], self.pk.S2.values[j])
                prod /= self.rlc(self.C.values[j], self.pk.S3.values[j])
            Z_values.append(prod)

        # Check that the last term Z_n = 1
        # NOTE: assertion shouldn't actually be true, I've commented it out
        # assert Z_values.pop() == 1

        # Sanity-check that Z was computed correctly
        for i in range(group_order):
            assert (
                self.rlc(self.A.values[i], roots_of_unity[i])
                * self.rlc(self.B.values[i], 2 * roots_of_unity[i])
                * self.rlc(self.C.values[i], 3 * roots_of_unity[i])
            ) * Z_values[i] - (
                self.rlc(self.A.values[i], self.pk.S1.values[i])
                * self.rlc(self.B.values[i], self.pk.S2.values[i])
                * self.rlc(self.C.values[i], self.pk.S3.values[i])
            ) * Z_values[
                (i + 1) % group_order
            ] == 0


        # Construct Z, Lagrange interpolation polynomial for Z_values
        # Cpmpute z_1 commitment to Z polynomial
        self.Z = Polynomial(Z_values, Basis.LAGRANGE)

        # Return z_1
        return Message2(self.setup.commit(self.Z))

    def round_3(self) -> Message3:
        group_order = self.group_order
        setup = self.setup

        # Compute the quotient polynomial

        # List of roots of unity at 4x fineness, i.e. the powers of µ
        # where µ^(4n) = 1
        fine_roots = Scalar.roots_of_unity(4*group_order)
        # Using self.fft_expand, move A, B, C into coset extended Lagrange basis


        # Expand public inputs polynomial PI into coset extended Lagrange


        # Expand selector polynomials pk.QL, pk.QR, pk.QM, pk.QO, pk.QC
        # into the coset extended Lagrange basis


        # Expand permutation grand product polynomial Z into coset extended
        # Lagrange basis
        A, B, C, PI, QL, QR, QM, QO, QC, Z, Zshift, S1, S2, S3 = [
            self.fft_expand(P) for P in [
                self.A, self.B, self.C, self.PI,
                self.pk.QL, self.pk.QR, self.pk.QM, self.pk.QO, self.pk.QC,
                self.Z, self.Z.shift(1), self.pk.S1, self.pk.S2, self.pk.S3
                ]
        ]
        # Expand shifted Z(ω) into coset extended Lagrange basis

        # Expand permutation polynomials pk.S1, pk.S2, pk.S3 into coset
        # extended Lagrange basis

        # Compute Z_H = X^N - 1, also in evaluation form in the coset
        ZH = Polynomial([Scalar(-1)] + [Scalar(0)] * (group_order - 1) + [self.fft_cofactor ** group_order] + [Scalar(0)] * (3*group_order - 1), Basis.MONOMIAL).fft()
        # Compute L0, the Lagrange basis polynomial that evaluates to 1 at x = 1 = ω^0
        # and 0 at other roots of unity

        # Expand L0 into the coset extended Lagrange basis
        self.L0 = self.fft_expand(
            Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE)
        )

        # Compute the quotient polynomial (called T(x) in the paper)
        # It is only possible to construct this polynomial if the following
        # equations are true at all roots of unity {1, w ... w^(n-1)}:
        # 1. All gates are correct:
        #    A * QL + B * QR + A * B * QM + C * QO + PI + QC = 0
        #
        # 2. The permutation accumulator is valid:
        #    Z(wx) = Z(x) * (rlc of A, X, 1) * (rlc of B, 2X, 1) *
        #                   (rlc of C, 3X, 1) / (rlc of A, S1, 1) /
        #                   (rlc of B, S2, 1) / (rlc of C, S3, 1)
        #    rlc = random linear combination: term_1 + beta * term2 + gamma * term3
        #
        # 3. The permutation accumulator equals 1 at the start point
        #    (Z - 1) * L0 = 0
        #    L0 = Lagrange polynomial, equal at all roots of unity except 1
        X = Polynomial([Scalar(0)] + [Scalar(1)] + [Scalar(0)] * (group_order - 2), Basis.MONOMIAL).fft()

        X = self.fft_expand(X)
        BIG1 = A*B*QM + A*QL + B*QR + C*QO + PI + QC
        BIG2 = Z * (A + X * self.beta + self.gamma) * (B + X * (2 * self.beta) + self.gamma) * (C + X * (3 * self.beta) + self.gamma) * self.alpha
        BIG2 -= Zshift * (A + S1 * self.beta + self.gamma) * (B + S2 * self.beta + self.gamma) * (C + S3 * self.beta + self.gamma) * self.alpha
        BIG3 = (Z - Scalar(1)) * self.L0 * (self.alpha ** 2)

        BIG = BIG1 + BIG2 + BIG3

        QUOT = BIG / ZH

        # Sanity check: QUOT has degree < 3n
        assert (
            self.expanded_evals_to_coeffs(BIG1/ZH).values[-group_order:]
            == [0] * group_order
        )
        assert (
            self.expanded_evals_to_coeffs(BIG2/ZH).values[-group_order:]
            == [0] * group_order
        )
        assert (
            self.expanded_evals_to_coeffs(BIG3/ZH).values[-group_order:]
            == [0] * group_order
        )
        print("Generated the quotient polynomial")

        # Split up T into T1, T2 and T3 (needed because T has degree 3n - 4, so is
        # too big for the trusted setup)

        T = QUOT.ifft()
        self.T1 = self.expanded_evals_to_coeffs(Polynomial(T.values[0:group_order], Basis.MONOMIAL).fft()).fft()
        self.T2 = self.expanded_evals_to_coeffs(Polynomial(T.values[group_order:2*group_order], Basis.MONOMIAL).fft()).fft() / (self.fft_cofactor ** group_order)
        self.T3 = self.expanded_evals_to_coeffs(Polynomial(T.values[2*group_order:3*group_order], Basis.MONOMIAL).fft()).fft() / (self.fft_cofactor ** (group_order * 2))

        # Sanity check that we've computed T1, T2, T3 correctly
        assert (
            self.T1.barycentric_eval(self.fft_cofactor)
            + self.T2.barycentric_eval(self.fft_cofactor) * self.fft_cofactor**group_order
            + self.T3.barycentric_eval(self.fft_cofactor) * self.fft_cofactor ** (group_order * 2)
        ) == QUOT.values[0]

        print("Generated T1, T2, T3 polynomials")

        # Compute commitments t_lo_1, t_mid_1, t_hi_1 to T1, T2, T3 polynomials
        t_lo_1 = setup.commit(self.T1)
        t_mid_1 = setup.commit(self.T2)
        t_hi_1 = setup.commit(self.T3)
        # Return t_lo_1, t_mid_1, t_hi_1
        return Message3(t_lo_1, t_mid_1, t_hi_1)

    def round_4(self) -> Message4:
        # Compute evaluations to be used in constructing the linearization polynomial.

        # Compute a_eval = A(zeta)
        # Compute b_eval = B(zeta)
        # Compute c_eval = C(zeta)
        # Compute s1_eval = pk.S1(zeta)
        # Compute s2_eval = pk.S2(zeta)
        # Compute z_shifted_eval = Z(zeta * ω)
        self.a_eval, self.b_eval, self.c_eval, self.s1_eval, self.s2_eval, self.z_shifted_eval = (
            P.barycentric_eval(self.zeta) for P in [
                self.A, self.B, self.C, self.pk.S1, self.pk.S2, self.Z.shift(1)
            ]
        )

        # Return a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval
        return Message4(self.a_eval, self.b_eval, self.c_eval, self.s1_eval, self.s2_eval, self.z_shifted_eval)

    def round_5(self) -> Message5:
        # Evaluate the Lagrange basis polynomial L0 at zeta
        # Evaluate the vanishing polynomial Z_H(X) = X^n - 1 at zeta
        l0_eval = self.expanded_evals_to_coeffs(self.L0).fft().barycentric_eval(self.zeta)
        zh_eval = self.zeta ** self.group_order - 1
        # Move T1, T2, T3 into the coset extended Lagrange basis
        # Move pk.QL, pk.QR, pk.QM, pk.QO, pk.QC into the coset extended Lagrange basis
        # Move Z into the coset extended Lagrange basis
        # Move pk.S3 into the coset extended Lagrange basis

        T1, T2, T3, QL, QR, QM, QO, QC, Z, S3 = [
            self.fft_expand(P) for P in [
                self.T1, self.T2, self.T3,
                self.pk.QL, self.pk.QR, self.pk.QM, self.pk.QO, self.pk.QC,
                self.Z, self.pk.S3
            ]
        ]
        # Compute the "linearization polynomial" R. This is a clever way to avoid
        # needing to provide evaluations of _all_ the polynomials that we are
        # checking an equation betweeen: instead, we can "skip" the first
        # multiplicand in each term. The idea is that we construct a
        # polynomial which is constructed to equal 0 at Z only if the equations
        # that we are checking are correct, and which the verifier can reconstruct
        # the KZG commitment to, and we provide proofs to verify that it actually
        # equals 0 at Z
        #
        # In order for the verifier to be able to reconstruct the commitment to R,
        # it has to be "linear" in the proof items, hence why we can only use each
        # proof item once; any further multiplicands in each term need to be
        # replaced with their evaluations at Z, which do still need to be provided

        BIG1 = QM * (self.a_eval * self.b_eval) + QL * self.a_eval + QR * self.b_eval + QO * self.c_eval + self.fft_expand(self.PI) + QC
        BIG2 = Z * (self.a_eval + self.zeta * self.beta + self.gamma) * (self.b_eval + self.zeta * (2 * self.beta) + self.gamma) * (self.c_eval + self.zeta * (3 * self.beta) + self.gamma) * self.alpha
        BIG2 -= (S3 * self.beta + self.gamma + self.c_eval) * self.z_shifted_eval * (self.a_eval + self.s1_eval * self.beta + self.gamma) * (self.b_eval + self.s2_eval * self.beta + self.gamma) * self.alpha
        BIG3 = (Z - Scalar(1)) * l0_eval * (self.alpha ** 2)
        BIG4 = (T1 + T2 * (self.zeta ** self.group_order) + T3 * (self.zeta ** (2*self.group_order))) * zh_eval

        
        R = BIG1 + BIG2 + BIG3 - BIG4
        


        # Commit to R

        r_commit = self.setup.commit(R)

        # Sanity-check R
        assert self.expanded_evals_to_coeffs(R).fft().barycentric_eval(self.zeta) == 0

        print("Generated linearization polynomial R")

        # Generate proof that W(z) = 0 and that the provided evaluations of
        # A, B, C, S1, S2 are correct

        # Move A, B, C into the coset extended Lagrange basis
        # Move pk.S1, pk.S2 into the coset extended Lagrange basis
        A, B, C, S1, S2 = [
            self.fft_expand(P) for P in [
                self.A, self.B, self.C,
                self.pk.S1, self.pk.S2
            ]
        ]
        # In the COSET EXTENDED LAGRANGE BASIS,
        # Construct W_Z = (
        #     R
        #   + v * (A - a_eval)
        #   + v**2 * (B - b_eval)
        #   + v**3 * (C - c_eval)
        #   + v**4 * (S1 - s1_eval)
        #   + v**5 * (S2 - s2_eval)
        # ) / (X - zeta)
        X = Polynomial([Scalar(0)] + [Scalar(1)] + [Scalar(0)] * (self.group_order - 2), Basis.MONOMIAL).fft()

        X = self.fft_expand(X)

        W_z = R
        W_z += (A - self.a_eval) * self.v
        W_z += (B - self.b_eval) * (self.v ** 2)
        W_z += (C - self.c_eval) * (self.v ** 3)
        W_z += (S1 - self.s1_eval) * (self.v ** 4)
        W_z += (S2 - self.s2_eval) * (self.v ** 5)
        W_z /= (X - self.zeta)
        W_z = self.expanded_evals_to_coeffs(W_z)

        # Check that degree of W_z is not greater than n
        assert W_z.values[self.group_order:] == [0] * (self.group_order * 3)

        # Compute W_z_1 commitment to W_z

        W_z_1 = self.setup.commit(W_z.fft())

        # Generate proof that the provided evaluation of Z(z*w) is correct. This
        # awkwardly different term is needed because the permutation accumulator
        # polynomial Z is the one place where we have to check between adjacent
        # coordinates, and not just within one coordinate.
        # In other words: Compute W_zw = (Z - z_shifted_eval) / (X - zeta * ω)

        W_zw = (Z - self.z_shifted_eval) / (X - self.zeta * Scalar.root_of_unity(self.group_order))
        W_zw = self.expanded_evals_to_coeffs(W_zw)
        # Check that degree of W_z is not greater than n
        assert W_zw.values[self.group_order:] == [0] * (self.group_order * 3)

        # Compute W_z_1 commitment to W_z
        W_zw_1 = self.setup.commit(W_zw.fft())
        print("Generated final quotient witness polynomials")

        # Return W_z_1, W_zw_1
        return Message5(W_z_1, W_zw_1)

    def fft_expand(self, x: Polynomial):
        return x.to_coset_extended_lagrange(self.fft_cofactor)

    def expanded_evals_to_coeffs(self, x: Polynomial):
        return x.coset_extended_lagrange_to_coeffs(self.fft_cofactor)

    def rlc(self, term_1, term_2):
        return term_1 + term_2 * self.beta + self.gamma
