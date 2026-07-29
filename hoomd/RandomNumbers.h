// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
   \file RandomNumbers.h
   \brief Declaration of hoomd::RandomNumbers

   This header includes templated generators for various types of random numbers required used
   throughout hoomd. These work with the RandomGenerator generator that wraps OpenRAND's Philox4x32
   RNG with an API that handles streams of random numbers originated from a seed.
 */

#pragma once

#include "HOOMDMath.h"

#include <hoomd/extern/OpenRAND/include/openrand/philox.h>

#include <limits>
#include <type_traits>

#ifdef __HIPCC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif // __HIPCC__

namespace util
    {
using std::make_signed;
using std::make_unsigned;

#if defined(__HIPCC__) || defined(_LIBCPP_HAS_NO_CONSTEXPR)

// Amazing! cuda thinks numeric_limits::max() is a __host__ function, so
// we can't use it in a device function.
//
// The LIBCPP_HAS_NO_CONSTEXP test catches situations where the libc++
// library thinks that the compiler doesn't support constexpr, but we
// think it does.  As a consequence, the library declares
// numeric_limits::max without constexpr.  This workaround should only
// affect a narrow range of compiler/library pairings.
//
// In both cases, we find max() by computing ~(unsigned)0 right-shifted
// by is_signed.
template<typename T> constexpr static inline DEVICE T maxTvalue()
    {
    typedef typename make_unsigned<T>::type uT;
    return (~uT(0)) >> std::numeric_limits<T>::is_signed;
    }
#else
template<typename T> constexpr static inline T maxTvalue()
    {
    return std::numeric_limits<T>::max();
    }
#endif

// u01: Input is a W-bit integer (signed or unsigned).  It is cast to
//   a W-bit unsigned integer, multiplied by Ftype(2^-W) and added to
//   Ftype(2^(-W-1)).  A good compiler should optimize it down to an
//   int-to-float conversion followed by a multiply and an add, which
//   might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  result is a uniformly distributed floating point number in [0, 1].
//  The result is never exactly 0.0.
//  The smallest value returned is 2^-W.
//  Let M be the number of mantissa bits in Ftype.
//  If W>M  then the largest value retured is 1.0.
//  If W<=M then the largest value returned is the largest Ftype less than 1.0.
template<typename Ftype, typename Itype> static inline DEVICE Ftype u01(Itype in)
    {
    typedef typename make_unsigned<Itype>::type Utype;
    constexpr Ftype factor = Ftype(1.) / (Ftype(maxTvalue<Utype>()) + Ftype(1.));
    constexpr Ftype halffactor = Ftype(0.5) * factor;
    return Ftype(Utype(in)) * factor + halffactor;
    }

// uneg11: Input is a W-bit integer (signed or unsigned).  It is cast
//    to a W-bit signed integer, multiplied by Ftype(2^-(W-1)) and
//    then added to Ftype(2^(-W-2)).  A good compiler should optimize
//    it down to an int-to-float conversion followed by a multiply and
//    an add, which might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  output is a uniformly distributed floating point number in [-1, 1].
//  The result is never exactly 0.0.
//  The smallest absolute value returned is 2^-(W-1)
//  Let M be the number of mantissa bits in Ftype.
//  If W>M  then the largest value retured is 1.0 and the smallest is -1.0.
//  If W<=M then the largest value returned is the largest Ftype less than 1.0
//    and the smallest value returned is the smallest Ftype greater than -1.0.
template<typename Ftype, typename Itype> static inline DEVICE Ftype uneg11(Itype in)
    {
    typedef typename make_signed<Itype>::type Stype;
    constexpr Ftype factor = Ftype(1.) / (Ftype(maxTvalue<Stype>()) + Ftype(1.));
    constexpr Ftype halffactor = Ftype(0.5) * factor;
    return Ftype(Stype(in)) * factor + halffactor;
    }

// end code copied from random123 examples
    } // namespace util

namespace hoomd
    {
/** RNG seed

    RandomGenerator initializes with a 64-bit seed and a 128-bit counter. Seed and Counter provide
    interfaces for common seeding patterns used across HOOMD to prevent code duplication and help
    ensure that seeds are initialized correctly.

    Seed provides one constructor as we expect this to be used everywhere in HOOMD. The constructor
    is a function of the class id, the current timestep and the user seed.
*/
class Seed
    {
    public:
    /** Construct a Seed from a class ID, timestep and user seed.

        The seed is 8 bytes. Construct this from an 1 byte class id, 2 byte seed, and the lower
        5 bytes of the timestep.

        When multiple class instances can instantiate RandomGenerator objects, include values in the
        Counter that are unique to each instance. Otherwise the separate instances will generate
        identical sequences of random numbers.

        Code inside HOOMD should name the class ID value in RNGIdentifiers.h and ensure that each
        class has a unique id. External plugins should use values 200 or larger.

        id seed1 seed0 timestep4 | timestep3 timestep2 timestep1 timestep0
    */
    DEVICE Seed(uint8_t id, uint64_t timestep, uint16_t seed)
        {
        m_key = static_cast<uint64_t>(id) << 56 | static_cast<uint64_t>(seed) << 40
                | static_cast<uint64_t>(timestep & 0x000000ffffffffff);
        }

    /// Get the key
    DEVICE const uint64_t& getKey() const
        {
        return m_key;
        }

    private:
    uint64_t m_key;
    };

/** RNG Counter

    Counter provides a number of constructors that support a variety of seeding needs throughought
    HOOMD.
*/
class Counter
    {
    public:
    /** Default constructor.

    Constructs a 0 valued counter.
    */
    DEVICE Counter(uint32_t _a = 0, uint32_t _b = 0, uint32_t _c = 0) : a(_a), b(_b), c(_c) { }

    /// Get the counter
    DEVICE std::array<uint32_t, 4> getCounter() const
        {
        return {0, c, b, a};
        }

    const uint32_t a, b, c;
    };

//! Philox random number generator
/*! Philix is a counter based random number generator. Given an input seed vector,
     it produces a random output. Outputs from one seed to the next are not correlated.
     This class implements a convenience API around OpenRAND that allows short streams
     (less than 2**32-1) of random numbers starting from a given Seed and Counter.

     Internally, we use the philox 4x32 RNG from OpenRAND, The first two seeds map to the
     key and the remaining seeds map to the counter. One element from the counter is used
     to generate the stream of values. Constructors provide ways to conveniently initialize
     the RNG with any number of seeds or counters.

     Counter based RNGs are useful for MD simulations: See

     C.L. Phillips, J.A. Anderson, and S.C. Glotzer. "Pseudo-random number generation
     for Brownian Dynamics and Dissipative Particle Dynamics simulations on GPU devices",
     J. Comput. Phys. 230, 7191-7201 (2011).

     and

     Y. Afshar, F. Schmid, A. Pishevar, and S. Worley. "Exploiting seeding of random
     number generators for efficient domain decomposition parallelization of dissipative
     particle dynamics", Comput. Phys. Commun. 184, 1119-1128 (2013).

     for more details.
 */
class RandomGenerator
    {
    public:
    /** Construct a random generator from a Seed and a Counter

        @param seed RNG seed.
        @param counter Initial value of the RNG counter.
    */
    DEVICE inline RandomGenerator(const Seed& seed, const Counter& counter);

    //! Generate a uniform random uint32_t
    DEVICE inline uint32_t generate_u32()
        {
        return rng.rand<uint32_t>();
        }

    //! Generate a uniform random uint64_t
    DEVICE inline uint64_t generate_u64()
        {
        return rng.rand<uint64_t>();
        }

    //! Generate two uniform random uint64_t
    /*! \param out1 [out] A random uniform 64-bit unsigned integer.
        \param out2 [out] A random uniform 64-bit unsigned integer.
     */
    DEVICE inline void generate_2u64(uint64_t& out1, uint64_t& out2)
        {
        out1 = rng.rand<uint64_t>();
        out2 = rng.rand<uint64_t>();
        }

    template<class Real> DEVICE inline Real generate_canonical()
        {
        return util::u01<Real>(generate_u64());
        }

    private:
    /// The random number generator
    openrand::Philox rng;
    };

DEVICE inline RandomGenerator::RandomGenerator(const Seed& seed, const Counter& counter)
    : rng(seed.getKey(), counter.a, counter.b, counter.c)
    {
    }

/** HOOMD 2.x-compatible Philox4x32-10 random number generator.

    HOOMD 2.9.7 hashes the user seed, stores the 32-bit class identifier and
    hashed seed in the two Philox key words, and constructs the counter as
    ``{draw_index, 0, timestep, particle_tag}``. OpenRAND uses a different word
    layout. This generator implements the legacy layout directly so selected
    integration methods can reproduce a HOOMD 2.x stream without changing the
    default HOOMD 7 stream.

    The timestep truncation to 32 bits is intentional: HOOMD 2.9.7 accepted a
    32-bit timestep in the corresponding integration method.
 */
class LegacyRandomGenerator
    {
    public:
    /** Construct a legacy stream.

        @param identifier HOOMD 2.x 32-bit RNG class identifier.
        @param hashed_seed User seed after applying `hashUserSeed`.
        @param particle_tag Global particle tag.
        @param timestep Current timestep.
     */
    DEVICE inline LegacyRandomGenerator(uint32_t identifier,
                                        uint32_t hashed_seed,
                                        uint32_t particle_tag,
                                        uint64_t timestep)
        : m_key0(identifier), m_key1(hashed_seed), m_particle_tag(particle_tag),
          m_timestep(static_cast<uint32_t>(timestep)), m_draw_index(0)
        {
        }

    //! Apply the seed transformation used by HOOMD 2.9.7 Langevin methods.
    DEVICE static inline uint32_t hashUserSeed(uint32_t seed)
        {
        seed = seed * uint32_t(0x12345677) + uint32_t(0x12345);
        seed ^= seed >> 16;
        seed *= uint32_t(0x45679);
        return seed;
        }

    //! Generate a uniform random uint32_t.
    DEVICE inline uint32_t generate_u32()
        {
        uint32_t output[4];
        generate(output);
        return output[0];
        }

    //! Generate a uniform random uint64_t.
    DEVICE inline uint64_t generate_u64()
        {
        uint32_t output[4];
        generate(output);
        return (static_cast<uint64_t>(output[0]) << 32) | static_cast<uint64_t>(output[1]);
        }

    //! Generate two uniform random uint64_t values from one Philox block.
    DEVICE inline void generate_2u64(uint64_t& out1, uint64_t& out2)
        {
        uint32_t output[4];
        generate(output);
        out1 = (static_cast<uint64_t>(output[0]) << 32) | static_cast<uint64_t>(output[1]);
        out2 = (static_cast<uint64_t>(output[2]) << 32) | static_cast<uint64_t>(output[3]);
        }

    template<class Real> DEVICE inline Real generate_canonical()
        {
        return util::u01<Real>(generate_u64());
        }

    private:
    DEVICE static inline uint32_t multiplyLow(uint32_t left, uint32_t right, uint32_t& high)
        {
        const uint64_t product = static_cast<uint64_t>(left) * static_cast<uint64_t>(right);
        high = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product);
        }

    DEVICE static inline void round(const uint32_t key[2], uint32_t counter[4])
        {
        uint32_t high0;
        uint32_t high1;
        const uint32_t low0 = multiplyLow(uint32_t(0xD2511F53), counter[0], high0);
        const uint32_t low1 = multiplyLow(uint32_t(0xCD9E8D57), counter[2], high1);
        counter[0] = high1 ^ counter[1] ^ key[0];
        counter[1] = low1;
        counter[2] = high0 ^ counter[3] ^ key[1];
        counter[3] = low0;
        }

    DEVICE inline void generate(uint32_t output[4])
        {
        uint32_t key[2] = {m_key0, m_key1};
        output[0] = m_draw_index;
        output[1] = 0;
        output[2] = m_timestep;
        output[3] = m_particle_tag;

        for (unsigned int iteration = 0; iteration < 10; ++iteration)
            {
            if (iteration > 0)
                {
                key[0] += uint32_t(0x9E3779B9);
                key[1] += uint32_t(0xBB67AE85);
                }
            round(key, output);
            }
        ++m_draw_index;
        }

    const uint32_t m_key0;
    const uint32_t m_key1;
    const uint32_t m_particle_tag;
    const uint32_t m_timestep;
    uint32_t m_draw_index;
    };

namespace detail
    {
//! Generate a random value in [2**(-65), 1]
/*!
    \returns A random uniform float in [2**(-65), 1]

    \post The state of the generator is advanced one step.
 */
template<class Real> DEVICE inline Real generate_canonical(RandomGenerator& rng)
    {
    return rng.generate_canonical<Real>();
    }

//! Generate a legacy-compatible random value in [2**(-65), 1].
template<class Real> DEVICE inline Real generate_canonical(LegacyRandomGenerator& rng)
    {
    return rng.generate_canonical<Real>();
    }
    } // namespace detail

//! Generate a uniform random value in [a,b]
/*! For all practical purposes, the range returned by this function is [a,b]. This is due to round
   off error: e.g. for a=1.0, 1.0+2**(-65) == 1.0. For small values of a, the range may become
   (a,b]. It depends on the round off that occurs in a + (b-a)*u, where u is in the range [2**(-65),
   1].
*/
template<typename Real> class UniformDistribution
    {
    public:
    //! Constructor
    /*! \param _a Left end point of the interval
        \param _b Right end point of the interval
    */
    DEVICE explicit UniformDistribution(Real _a = Real(0.0), Real _b = Real(1.0))
        : a(_a), width(_b - _a)
        {
        }

    //! Draw a value from the distribution
    /*! \param rng Random number generator
        \returns uniform random value in [a,b]
    */
    template<typename RNG> DEVICE inline Real operator()(RNG& rng)
        {
        return a + width * detail::generate_canonical<Real>(rng);
        }

    private:
    const Real a;     //!< Left end point of the interval
    const Real width; //!< Width of the interval
    };

//! Generate normally distributed random values
/*! Use the Box-Muller method to generate normally distributed random values.
 */
template<typename Real> class NormalDistribution
    {
    public:
    //! Constructor
    /*! \param _sigma Standard deviation of the distribution
        \param _mu Mean of the distribution
    */
    DEVICE explicit NormalDistribution(Real _sigma = Real(1.0), Real _mu = Real(0.0))
        : sigma(_sigma), mu(_mu)
        {
        }

    //! Draw a value from the distribution
    /*! \param rng Random number generator
        \returns normally distributed random value in with standard deviation *sigma* and mean *mu*.
    */
    template<typename RNG> DEVICE inline Real operator()(RNG& rng)
        {
        uint64_t u0, u1;
        rng.generate_2u64(u0, u1);

        // from random123/examples/boxmuller.hpp
        Real x, y;
        fast::sincospi(util::uneg11<Real>(u0), x, y);
        Real r = fast::sqrt(Real(-2.0)
                            * fast::log(util::u01<Real>(u1))); // u01 is guaranteed to avoid 0.
        x *= r;
        return x * sigma + mu;
        }

    //! Draw two values from the distribution
    /*! \param out1 [out] First output
        \param out2 [out] Second output
        \param rng Random number generator
        \returns normally distributed random value in with standard deviation *sigma* and mean *mu*.
    */
    template<typename RNG> DEVICE inline void operator()(Real& out1, Real& out2, RNG& rng)
        {
        uint64_t u0, u1;
        rng.generate_2u64(u0, u1);

        // from random123/examples/boxmuller.hpp
        Real x, y;
        fast::sincospi(util::uneg11<Real>(u0), x, y);
        Real r = fast::sqrt(Real(-2.0)
                            * fast::log(util::u01<Real>(u1))); // u01 is guaranteed to avoid 0.
        r = r * sigma;
        x *= r;
        y *= r;
        out1 = x + mu;
        out2 = y + mu;
        }

    private:
    const Real sigma; //!< Standard deviation
    const Real mu;    //!< Mean
    };

//! Generate random points on the surface of a sphere
template<typename Real> class SpherePointGenerator
    {
    public:
    DEVICE explicit SpherePointGenerator() { }

    template<typename RNG, typename Real3> DEVICE inline void operator()(RNG& rng, Real3& point)
        {
        // draw a random angle
        const Real theta = UniformDistribution<Real>(Real(0), Real(2.0 * M_PI))(rng);

        // draw u (should typically only happen once) ensuring that
        // 1-u^2 > 0 so that the square-root is defined
        Real u, one_minus_u2;
        do
            {
            u = UniformDistribution<Real>(Real(-1.0), Real(1.0))(rng);
            one_minus_u2 = Real(1.0) - u * u;
            } while (one_minus_u2 < Real(0.0));

        // project onto the sphere surface
        const Real sqrtu = fast::sqrt(one_minus_u2);
        fast::sincos(theta, (Real&)point.y, (Real&)point.x);
        point.x *= sqrtu;
        point.y *= sqrtu;
        point.z = u;
        }
    };

//! Generator for gamma-distributed random variables
/*!
 * The probability distribution function is:
 *
 * \f[
 *    p(x) = \frac{1}{\Gamma(\alpha) b^\alpha} x^{\alpha - 1} e^{-x/b}
 * \f]
 *
 * which has parameters \f$\alpha\f$ and \a b. In the Maxwell-Boltzmann
 * distribution, \a x is the kinetic energy of \a N particles, \a b plays the
 * role of the thermal energy \a kT , and \f$\alpha = d(N-1)/2\f$ for
 * dimensionality \a d.
 *
 * The method used to generate the random numbers is a rejection-sampling method:
 *
 *      "A simple method for generating gamma variables", ACM Transactions on
 *      Mathematical Software (TOMS), vol. 26, issue 3, Sept. 2000, 363--372.
 *      https://doi.org/10.1145/358407.358414
 *
 * \tparam Real Precision of the random number
 */
template<typename Real> class GammaDistribution
    {
    public:
    //! Constructor
    /*!
     * \param alpha
     * \param b
     */
    DEVICE explicit GammaDistribution(const Real alpha, const Real b) : m_alpha(alpha), m_b(b) { }

    //! Draw a random number from the gamma distribution
    /*!
     * \param rng Philox random number generator
     * \returns A gamma distributed random variate
     *
     * The implementation of this method is inspired by that of the GSL,
     * and also as discussed online:
     *
     *      https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
     *
     * The squeeze test is performed to bypass some transcendental calls.
     */
    template<typename RNG> DEVICE inline Real operator()(RNG& rng)
        {
        if (m_alpha <= 0)
            {
#ifndef __HIPCC__
            throw std::domain_error("alpha must be positive.");
#else
            return 0;
#endif
            }

        Real alpha = m_alpha;
        if (m_alpha < Real(1.0))
            {
            // when m_alpha < 1, handle specially (see below)
            alpha += Real(1.0);
            }
        Real d = alpha - Real(1. / 3.);
        Real c = fast::rsqrt(d) / Real(3.);

        Real v;
        while (1)
            {
            // first draw a valid Marsaglia v value using the normal distribution
            Real x;
            do
                {
                x = m_normal(rng);
                v = Real(1.0) + c * x;
                } while (v <= Real(0.));
            v = v * v * v;

            // draw uniform and perform cheap squeeze test first
            const Real x2 = x * x;
            Real u = detail::generate_canonical<Real>(rng);
            if (u < Real(1.0) - Real(0.0331) * x2 * x2)
                break;

            // otherwise, do expensive log comparison
            if (fast::log(u) < Real(0.5) * x2 + d * (Real(1.0) - v + fast::log(v)))
                break;
            }

        // convert the Gamma(alpha,1) to Gamma(alpha,b)
        v *= d * m_b;
        if (m_alpha < Real(1.0))
            {
            // finish special case
            v *= fast::pow(detail::generate_canonical<Real>(rng), Real(1.0) / m_alpha);
            }
        return v;
        }

    private:
    Real m_alpha; //!< Gamma distribution alpha parameter
    Real m_b;     //!< Gamma-distribution b-parameter

    NormalDistribution<Real> m_normal; //!< Normal variate generator
    };

//! Generate uniform random unsigned integers in the range [0,m]
/*! This distribution is useful when selecting from a number of finite choices.
 */
class UniformIntDistribution
    {
    public:
    //! Constructor
    /*! \param _m Maximum value this distribution will return
     */
    DEVICE explicit UniformIntDistribution(uint32_t _m) : m(_m) { }

    //! Draw a value from the distribution
    /*! \param rng RNG to utilize in the move
        \returns a random number 0 <= i <= m with uniform probability.

        **Method**

        First, round m+1 up to the next nearest power of two -> max2. Then draw random numbers in
       the range [0 ... max2) using 32-but random values and a bitwise and with max2-1. Return the
       first random number found in the range.
    */
    template<typename RNG> DEVICE inline uint32_t operator()(RNG& rng)
        {
        // handle degenerate case where m==0
        if (m == 0)
            return 0;

        // algorithm to round up to the nearest power of two from
        // https://en.wikipedia.org/wiki/Power_of_two
        unsigned int n = m + 1;
        n = n - 1;
        n = n | (n >> 1);
        n = n | (n >> 2);
        n = n | (n >> 4);
        n = n | (n >> 8);
        n = n | (n >> 16);
        // Note: leaving off the n = n + 1 because we are going to & with next highest power of 2 -1

        unsigned int result;
        do
            {
            result = rng.generate_u32() & n;
            } while (result > m);

        return result;
        }

    private:
    const uint32_t m; //!< Maximum value
    };

//! Generate Poisson distributed random values
/*! Use the method from:
   https://scicomp.stackexchange.com/questions/27330/how-to-generate-poisson-distributed-random-numbers-quickly-and-accurately/27334
    (code posted there is in the public domain)
*/
template<class Real> class PoissonDistribution
    {
    public:
    //! Constructor
    /*! \param _mean Distribution mean
     */
    DEVICE explicit PoissonDistribution(Real _mean) : mean(_mean) { }

    //! Draw a value from the distribution
    /*! \param rng Random number generator
        \returns normally Poisson distributed random number with mean *mean*.
    */
    template<typename RNG> DEVICE inline int operator()(RNG& rng)
        {
        // the value 13 is determined by empirical performance testing
        if (mean < 13)
            {
            return poissrnd_small(rng);
            }
        else
            {
            return poissrnd_large(rng);
            }
        }

    private:
    const Real mean; //!< Sample mean

    Real _lgamma(Real xx)
        {
        // code from /*! Use the method from:
        // https://scicomp.stackexchange.com/questions/27330/how-to-generate-poisson-distributed-random-numbers-quickly-and-accurately/27334
        // compute lgamma from series expansion
        Real pi = M_PI;
        Real xx2 = xx * xx;
        Real xx3 = xx2 * xx;
        Real xx5 = xx3 * xx2;
        Real xx7 = xx5 * xx2;
        Real xx9 = xx7 * xx2;
        Real xx11 = xx9 * xx2;
        return xx * fast::log(xx) - xx - Real(0.5) * fast::log(xx / (Real(2) * pi))
               + Real(1) / (Real(12) * xx) - Real(1) / (Real(360) * xx3)
               + Real(1) / (Real(1260) * xx5) - Real(1) / (Real(1680) * xx7)
               + Real(1) / (Real(1188) * xx9) - Real(691) / (Real(360360) * xx11);
        }

    template<typename RNG> DEVICE int poissrnd_small(RNG& rng)
        {
        Real L = fast::exp(-mean);
        Real p = 1;
        int result = 0;
        do
            {
            result++;
            p *= detail::generate_canonical<Real>(rng);
            } while (p > L);
        result--;
        return result;
        }

    template<typename RNG> DEVICE int poissrnd_large(RNG& rng)
        {
        Real r;
        Real x, m;
        Real pi = Real(M_PI);
        Real sqrt_mean = fast::sqrt(mean);
        Real log_mean = fast::log(mean);
        Real g_x;
        Real f_m;

        do
            {
            do
                {
                x = mean
                    + sqrt_mean
                          * slow::tan(pi * (detail::generate_canonical<Real>(rng) - Real(0.5)));
                } while (x < 0);
            g_x = sqrt_mean / (pi * ((x - mean) * (x - mean) + mean));
            m = slow::floor(x);
            f_m = fast::exp(m * log_mean - mean - lgamma(m + 1));
            r = f_m / g_x / Real(2.4);
            } while (detail::generate_canonical<Real>(rng) > r);
        return (int)m;
        }
    };

    } // end namespace hoomd
#undef DEVICE
