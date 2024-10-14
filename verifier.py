import py_ecc.bn128 as b
from utils import *
from dataclasses import dataclass
from curve import *
from transcript import Transcript
from poly import Polynomial, Basis


@dataclass
class VerificationKey:
    """Verification key"""

    # we set this to some power of 2 (so that we can FFT over it), that is at least the number of constraints we have (so we can Lagrange interpolate them)
    group_order: int
    # [q_M(x)]₁ (commitment to multiplication selector polynomial)
    Qm: G1Point
    # [q_L(x)]₁ (commitment to left selector polynomial)
    Ql: G1Point
    # [q_R(x)]₁ (commitment to right selector polynomial)
    Qr: G1Point
    # [q_O(x)]₁ (commitment to output selector polynomial)
    Qo: G1Point
    # [q_C(x)]₁ (commitment to constants selector polynomial)
    Qc: G1Point
    # [S_σ1(x)]₁ (commitment to the first permutation polynomial S_σ1(X))
    S1: G1Point
    # [S_σ2(x)]₁ (commitment to the second permutation polynomial S_σ2(X))
    S2: G1Point
    # [S_σ3(x)]₁ (commitment to the third permutation polynomial S_σ3(X))
    S3: G1Point
    # [x]₂ = xH, where H is a generator of G_2
    X_2: G2Point
    # nth root of unity (i.e. ω^1), where n is the program's group order.
    w: Scalar

    # More optimized version that tries hard to minimize pairings and
    # elliptic curve multiplications, but at the cost of being harder
    # to understand and mixing together a lot of the computations to
    # efficiently batch them
    def verify_proof(self, group_order: int, pf, public=[]) -> bool:
        # 4. Compute challenges
        beta, gamma, alpha, zeta, v, u = self.compute_challenges(pf)
        # 5. Compute zero polynomial evaluation Z_H(ζ) = ζ^n - 1
        zh_eval = zeta ** group_order - 1
        # 6. Compute Lagrange polynomial evaluation L_0(ζ)
        l0_eval = Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE).barycentric_eval(zeta)
        # 7. Compute public input polynomial evaluation PI(ζ).
        pi_eval = Polynomial([Scalar(-x) for x in public] + [Scalar(0)] * (group_order - len(public)), Basis.LAGRANGE).barycentric_eval(zeta)

        # Compute the constant term of R. This is not literally the degree-0
        # term of the R polynomial; rather, it's the portion of R that can
        # be computed directly, without resorting to elliptic cutve commitments
        proof = pf.flatten()
        def rlc(x, y):
            return x + beta * y + gamma
        # r_commit = ec_lincomb((
        #     (self.Qm, proof["a_eval"] * proof["b_eval"]),
        #     (self.Ql, proof["a_eval"]),
        #     (self.Qr, proof["b_eval"]),
        #     (self.Qo, proof["c_eval"]),
        #     (b.G1, pi_eval),
        #     (self.Qc, Scalar(1)),
        #     (proof["z_1"], alpha * rlc(proof["a_eval"], zeta) * rlc(proof["b_eval"], 2*zeta) * rlc(proof["c_eval"], 3*zeta)),
        #     (self.S3, - alpha * beta * proof["z_shifted_eval"] * rlc(proof["a_eval"], proof["s1_eval"]) * rlc(proof["b_eval"], proof["s2_eval"])),
        #     (b.G1, - alpha * (gamma + proof["c_eval"]) * proof["z_shifted_eval"] * rlc(proof["a_eval"], proof["s1_eval"]) * rlc(proof["b_eval"], proof["s2_eval"])),
        #     (proof["z_1"], (alpha ** 2) * l0_eval),
        #     (b.G1, - l0_eval * (alpha ** 2)),
        #     (proof["t_lo_1"], - zh_eval),
        #     (proof["t_mid_1"], - zh_eval * (zeta ** group_order)),
        #     (proof["t_hi_1"], - zh_eval * (zeta ** (2 * group_order)))
        # ))
        # print(r_commit)
        r0 = pi_eval 
        r0 += - alpha * (gamma + proof["c_eval"]) * proof["z_shifted_eval"] * rlc(proof["a_eval"], proof["s1_eval"]) * rlc(proof["b_eval"], proof["s2_eval"])
        r0 += - l0_eval * (alpha ** 2)
        # Compute D = (R - r0) + u * Z, and E and F

        # Let S(t) be the polynomial R(t) + v A(t) + v^2 B(t) + ...
        # We are trying to check that S(x) - S(zeta) = (x - zeta)W_z(x)
        # or equivalently S(x) - S(zeta) + zeta W_z(x) = x(W_z(x))
        # Similarly we are trying to check that Z(x) - Zshift(zeta) + w*zeta*W_zw(x) = x(W_zw(x))
        # In order to do this we take a random u and check that
        # S(x) - S(zeta) + zeta W_z(x) + u * (Z(x) - Zshift(zeta) + w*zeta*W_zw(x))
        #   = x(W_z(x) + u * W_zw(x))
        # To rewrite,
        # (S(x) - r0) + zeta W_z(x) + u Z(x) + u*w*zeta*W_zw(x) (let this be P1)
        # + r0 - S(zeta) - u*Zshift(zeta) (let this be P2)
        # = x(W_z(x) + u * W_zw(x))

        s_eval = proof["a_eval"] * v + proof["b_eval"] * v**2 + proof["c_eval"] * v**3 + proof["s1_eval"] * v**4 + proof["s2_eval"] * v**5
        p2 = r0 - s_eval - u* proof["z_shifted_eval"]
        P1 = ec_lincomb((
            (self.Qm, proof["a_eval"] * proof["b_eval"]),
            (self.Ql, proof["a_eval"]),
            (self.Qr, proof["b_eval"]),
            (self.Qo, proof["c_eval"]),
            (self.Qc, Scalar(1)),
            (proof["z_1"], (
                alpha * rlc(proof["a_eval"], zeta) * rlc(proof["b_eval"], 2*zeta) * rlc(proof["c_eval"], 3*zeta)) \
                + (alpha ** 2) * l0_eval \
                + u
            ),
            #(proof["z_1"], alpha * rlc(proof["a_eval"], zeta) * rlc(proof["b_eval"], 2*zeta) * rlc(proof["c_eval"], 3*zeta)),
            (self.S3, - alpha * beta * proof["z_shifted_eval"] * rlc(proof["a_eval"], proof["s1_eval"]) * rlc(proof["b_eval"], proof["s2_eval"])),
            #(proof["z_1"], (alpha ** 2) * l0_eval),
            (proof["t_lo_1"], - zh_eval),
            (proof["t_mid_1"], - zh_eval * (zeta ** group_order)),
            (proof["t_hi_1"], - zh_eval * (zeta ** (2 * group_order))),
            #(proof["z_1"], u),
            (proof["a_1"], v),
            (proof["b_1"], v**2),
            (proof["c_1"], v**3),
            (self.S1, v**4),
            (self.S2, v**5),
            (proof["W_z_1"], zeta),
            (proof["W_zw_1"], u*Scalar.root_of_unity(group_order)*zeta)
        ))
        # Run one pairing check to verify the last two checks.
        # What's going on here is a clever re-arrangement of terms to check
        # the same equations that are being checked in the basic version,
        # but in a way that minimizes the number of EC muls and even
        # compressed the two pairings into one. The 2 pairings -> 1 pairing
        # trick is basically to replace checking
        #
        # Y1 = A * (X - a) and Y2 = B * (X - b)
        #
        # with
        #
        # Y1 + A * a = A * X
        # Y2 + B * b = B * X
        #
        # so at this point we can take a random linear combination of the two
        # checks, and verify it with only one pairing.

        assert b.pairing(
            self.X_2,
            ec_lincomb((
                (proof["W_z_1"], Scalar(1)),
                (proof["W_zw_1"], u)
            ))
        ) == b.pairing(
            b.G2,
            ec_lincomb((
                (b.G1, p2),
                (P1, Scalar(1))
            ))
        )
        return True

    # Basic, easier-to-understand version of what's going on
    def verify_proof_unoptimized(self, group_order: int, pf, public=[]) -> bool:
        # 4. Compute challenges
        beta, gamma, alpha, zeta, v, u = self.compute_challenges(pf)
        # 5. Compute zero polynomial evaluation Z_H(ζ) = ζ^n - 1
        zh_eval = zeta ** group_order - 1
        # 6. Compute Lagrange polynomial evaluation L_0(ζ)
        l0_eval = Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE).barycentric_eval(zeta)
        # 7. Compute public input polynomial evaluation PI(ζ).
        pi_eval = Polynomial([Scalar(-x) for x in public] + [Scalar(0)] * (group_order - len(public)), Basis.LAGRANGE).barycentric_eval(zeta)
        # Recover the commitment to the linearization polynomial R,
        # exactly the same as what was created by the prover
        proof = pf.flatten()

        def rlc(x, y):
            return x + beta * y + gamma
        # BIG1 = QM * (self.a_eval * self.b_eval) + QL * self.a_eval + QR * self.b_eval + QO * self.c_eval + self.PI.barycentric_eval(self.zeta) + QC
        # BIG2 = Z * (self.a_eval + self.zeta * self.beta + self.gamma) * (self.b_eval + self.zeta * (2 * self.beta) + self.gamma) * (self.c_eval + self.zeta * (3 * self.beta) + self.gamma) * self.alpha
        # BIG2 -= (S3 * self.beta + self.gamma + self.c_eval) * self.z_shifted_eval * (self.a_eval + self.s1_eval * self.beta + self.gamma) * (self.b_eval + self.s2_eval * self.beta + self.gamma) * self.alpha
        # BIG3 = (Z - Scalar(1)) * l0_eval * (self.alpha ** 2)
        # BIG4 = (T1 + T2 * (self.zeta ** self.group_order) + T3 * (self.zeta ** (2*self.group_order))) * zh_eval
        r_commit = ec_lincomb((
            (self.Qm, proof["a_eval"] * proof["b_eval"]),
            (self.Ql, proof["a_eval"]),
            (self.Qr, proof["b_eval"]),
            (self.Qo, proof["c_eval"]),
            (b.G1, pi_eval),
            (self.Qc, Scalar(1)),
            (proof["z_1"], alpha * rlc(proof["a_eval"], zeta) * rlc(proof["b_eval"], 2*zeta) * rlc(proof["c_eval"], 3*zeta)),
            (self.S3, - alpha * beta * proof["z_shifted_eval"] * rlc(proof["a_eval"], proof["s1_eval"]) * rlc(proof["b_eval"], proof["s2_eval"])),
            (b.G1, - alpha * (gamma + proof["c_eval"]) * proof["z_shifted_eval"] * rlc(proof["a_eval"], proof["s1_eval"]) * rlc(proof["b_eval"], proof["s2_eval"])),
            (proof["z_1"], (alpha ** 2) * l0_eval),
            (b.G1, - l0_eval * (alpha ** 2)),
            (proof["t_lo_1"], - zh_eval),
            (proof["t_mid_1"], - zh_eval * (zeta ** group_order)),
            (proof["t_hi_1"], - zh_eval * (zeta ** (2 * group_order)))
        ))
        print(r_commit)
        

        # Verify that R(z) = 0 and the prover-provided evaluations
        # A(z), B(z), C(z), S1(z), S2(z) are all correct
        assert b.pairing(
            b.add(self.X_2, ec_mul(b.G2, -zeta)),
            proof["W_z_1"]
        ) == b.pairing(
            b.G2,
            ec_lincomb((
                (r_commit, Scalar(1)),
                (proof["a_1"], v),
                (b.G1, - proof["a_eval"] * v),
                (proof["b_1"], v**2),
                (b.G1, - proof["b_eval"] * v**2),
                (proof["c_1"], v**3),
                (b.G1, - proof["c_eval"] * v**3),
                (self.S1, v**4),
                (b.G1, - proof["s1_eval"] * v**4),
                (self.S2, v**5),
                (b.G1, - proof["s2_eval"] * v**5)
            ))
        )
        # Verify that the provided value of Z(zeta*w) is correct
        assert b.pairing(
            b.add(self.X_2, ec_mul(b.G2, -zeta*Scalar.root_of_unity(group_order))),
            proof["W_zw_1"]
        ) == b.pairing(
            b.G2,
            ec_lincomb((
                (proof["z_1"], Scalar(1)),
                (b.G1, - proof["z_shifted_eval"])
            ))
        )
        return True

    # Compute challenges (should be same as those computed by prover)
    def compute_challenges(
        self, proof
    ) -> tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
        transcript = Transcript(b"plonk")
        beta, gamma = transcript.round_1(proof.msg_1)
        alpha, _fft_cofactor = transcript.round_2(proof.msg_2)
        zeta = transcript.round_3(proof.msg_3)
        v = transcript.round_4(proof.msg_4)
        u = transcript.round_5(proof.msg_5)

        return beta, gamma, alpha, zeta, v, u
