const KEYCOUNT = Ref(1)
const SENTINEL = Int32[-1, -1, -1]
const COORD_MOVES = [1,1,2,2]
const MOVE_ORIENTATIONS = [+1,-1,+1,-1]
const SPECIES_ID_COUNTER = Ref(1)
const PKINK = 0.8
const ALPHABET = "abcdefghijklmnopqrstuvwxyz"

abstract type AbstractSpecies end
abstract type AbstractPolymer <: AbstractSpecies end
abstract type AbstractMonomer <: AbstractSpecies end
abstract type AbstractBulkMonomer <: AbstractMonomer end 
abstract type AbstractPolyMonomer <: AbstractMonomer end 
abstract type AbstractFastBulkMonomer <: AbstractBulkMonomer end 

export Species, Polymer, PolyMonomer, BulkMonomer, FastBulkMonomer, Lattice,
       RousePolymer, Parameters

######## Generic species #######
# MARK: -Generic

Base.@kwdef mutable struct Species <: AbstractSpecies
    id::Int32 = begin
      SPECIES_ID_COUNTER[] += 1
      SPECIES_ID_COUNTER[]
    end

    name::String = ""    # unique name of this species

    # required fields
    length::Union{Nothing, Int32} = nothing       # number of beads
    maxlength::Union{Nothing, Int32} = nothing   # max number of beads
    boxlength::Union{Nothing, Int32} = nothing    # box linear dimension
    pos::Union{Nothing, OrderedDict{Int32,Vector{Int32}}} = nothing   # xyz coordinates of all the beads
    head::Union{Nothing, Int32} = nothing
    latS::Union{Nothing, AbstractArray{Int32}} = nothing   # lattice
    tempVec::Union{Nothing, Vector{Int32}} = nothing
    nnIdxs::Union{Nothing, Vector{Int32}} = nothing
    freeNNs::Union{Nothing, Set{Int32}} = nothing
end


######## Polymer #######
# MARK: -Polymer

# Forward declare to allow PolyMonomer to reference it

######## Monomer (for bead-dependent interactions) #######
# MARK: -PolyMonomer
Base.@kwdef mutable struct PolyMonomer <: AbstractPolyMonomer
    id::Int32 = begin
        SPECIES_ID_COUNTER[] += 1
        SPECIES_ID_COUNTER[]
    end
    
    name::String = ""    # unique name of this Monomer

    # host polymer
    poly::AbstractPolymer
    
    # interaction potentials
    chemPot::Union{Nothing, Float64} = nothing   # chemical potential for this Monomer
    intPot::Union{Nothing, Float64} = nothing    # self-interaction potential
    
    # multi-Monomer interaction fields (mirrors AbstractBulkMonomer pattern)
    intSpecies::Union{Nothing, Vector{Tuple{AbstractMonomer, Int32}}} = nothing  # list of (Monomer, canonical index) pairs
    intMat::Union{Nothing, Matrix{Float64}} = nothing  # interaction matrix (symmetric, diagonal = self intPot)
    intIdx::Union{Nothing, Int32} = nothing  # this Monomer' canonical index in intMat
  end


# Now define Polymer completely
Base.@kwdef mutable struct Polymer <: AbstractPolymer
    
    id::Int32 = begin
      SPECIES_ID_COUNTER[] += 1
      SPECIES_ID_COUNTER[]
    end

    name::String = ""    # unique name of this species

    # required fields
    length::Int32        # number of beads
    maxlength::Int32     # max number of beads
    nMonomerTypes::Union{Nothing, Int32} = nothing  # number of distinct PolyMonomer

    # KMC base rates per move type (sets simulation time unit; unused by Metropolis)
    k0_add::Float64 = 1.0
    k0_kink::Float64 = 1.0
    k0_remove::Float64 = 1.0
    
    # key vector, position and initial, final bead positions
    boxlength::Union{Nothing, Int32} = nothing    # box linear dimension
    keys_vec::Union{Nothing, Vector{Int32}} = nothing  # pre-allocated key vector
    pos::Union{Nothing, OrderedDict{Int32, Vector{Int32}}} = nothing      # xyz coordinates of all the beads
    initPos::Union{Nothing, Vector{Int32}} = nothing     # anchor position on one side
    initShape::Union{Nothing, String} = nothing
    finalPos::Union{Nothing, Vector{Int32}} = nothing   # anchor position on the other side
    
    # lattice, ghost and partner species
    latS::Union{Nothing, AbstractArray{Int32}} = nothing   # lattice
    ghost::Union{Nothing, AbstractSpecies} = nothing
    
    # Monomer for bead-dependent interactions
    # Dictionary keyed by intIdx for O(1) lookup; pos[key][6] stores the intIdx
    monomers::Union{Nothing, Dict{Int32, PolyMonomer}} = nothing
    pattern::Symbol=:uniform
    customPattern::Union{Nothing, Vector{Int}}=nothing
    patternTails::Union{Nothing, Dict{Int32, Int32}} = nothing  # intIdx -> tail key for non-base types
    baseIntIdx::Union{Nothing, Int32} = nothing                  # intIdx of base type (can grow/shrink)
    
    # addBond!, removeBond!, identifyBond! variables
    head::Union{Nothing, Int32} = nothing
    ki::Union{Nothing, Int32} = nothing
    kj::Union{Nothing, Int32} = nothing
    kprev::Union{Nothing, Int32} = nothing
    knext::Union{Nothing, Int32} = nothing
    bi::Union{Nothing, Vector{Int32}} = nothing      # pre-allocated bond coordinate vectors
    bj::Union{Nothing, Vector{Int32}} = nothing
    bprev::Union{Nothing, Vector{Int32}} = nothing
    bnext::Union{Nothing, Vector{Int32}} = nothing
    bondVec::Union{Nothing, Vector{Int32}} = nothing       # direction vector for a given bond
    tempVec::Union{Nothing, Vector{Int32}} = nothing

    # moveTypes! variables
    moveVec::Union{Nothing, Vector{Int32}} = nothing       # move vector
    moveIdx::Union{Nothing, Int32} = nothing          # bead index to move next
    moveType::Union{Nothing, Int32} = nothing              # type of move: bond addition (1), bond removal (2) or kink (3)
    newPosi::Union{Nothing, Vector{Int32}} = nothing       # placeholder position (e.g. to store next bond position)
    newPosj::Union{Nothing, Vector{Int32}} = nothing       # placeholder position (e.g. to store next bond position)
    moveTypes::Union{Nothing, Vector{Int32}} = nothing     # array of possible move types around a given bead

    # proposeMove! variables
    moveTypeProbs::Union{Nothing, Vector{Float64}} = nothing   # move proposal probabilities
    nTypes::Union{Nothing, Vector{Float64}} = nothing          # how many of each move type there are
    hastingsFactor::Union{Nothing, Float64} = nothing
    
    # energy
    ham::Union{Nothing, Float64} = nothing       # Hamiltonian
    deltaH::Union{Nothing, Float64} = nothing    # Hamiltonian increment of a given move
    deltaHP::Union{Nothing, Float64} = nothing   # Hamiltonian increment from partner move
end


######## BulkMonomer #######
# MARK: -BulkMonomer

Base.@kwdef mutable struct BulkMonomer <: AbstractBulkMonomer
    
    id::Int32 = begin
      SPECIES_ID_COUNTER[] += 1
      SPECIES_ID_COUNTER[]
    end

    name::String = ""    # unique name of this species

    # required fields
    length::Int32        # number of beads

    chemPot::Union{Nothing,Float64} = nothing   # monomer chemical potential
    intPot::Union{Nothing,Float64} = nothing   # monomer interaction potential
    k0::Float64 = 1.0   # KMC base hop rate (sets simulation time unit)

    # key vector, position and initial, final bead positions
    boxlength::Union{Nothing, Int32} = nothing
    keys_vec::Union{Nothing, Vector{Int32}} = nothing  # pre-allocated key vector
    pos::Union{Nothing, OrderedDict{Int32,Vector{Int32}}} = nothing      # xyz coordinates of all the beads
    initPos::Union{Nothing, Vector{Int32}} = nothing     # anchor position on one side
    initShape::Union{Nothing, String} = nothing
    initParams::Union{Nothing, Int32} = nothing

    # lattice, ghost and partner species
    latS::Union{Nothing, AbstractArray{Int32}} = nothing
    ghost::Union{Nothing, AbstractSpecies} = nothing

    # multi-species interaction fields
    intSpecies::Union{Nothing, Vector{Tuple{AbstractMonomer, Int32}}} = nothing  # list of (species, canonical index) pairs
    intMat::Union{Nothing, Matrix{Float64}} = nothing  # interaction matrix (symmetric, diagonal = self intPot)
    intIdx::Union{Nothing, Int32} = nothing  # this species' canonical index in intMat

    # three-way coupling fields (BulkMonomer only)
    # couplingMat[i,j,k] = energy contribution when self (couplingIdx=i) overlaps context k and neighbors partner j
    # Symmetry: couplingMat[i,j,k] == couplingMat[i,k,j] (partner/context interchangeable)
    couplingSpecies::Union{Nothing, Vector{Tuple{AbstractBulkMonomer, Int32}}} = nothing  # list of (BulkMonomer, couplingIdx) pairs
    couplingMat::Union{Nothing, Array{Float64, 3}} = nothing  # m × n × n array (m = BulkMonomers, n = all species)
    couplingIdx::Union{Nothing, Int32} = nothing  # this BulkMonomer's index in couplingMat first dimension

    # moveTypes! variables
    bi::Union{Nothing, AbstractArray{Int32}} = nothing
    moveIdx::Union{Nothing, Int32} = nothing 
    moveVec::Union{Nothing, Vector{Int32}} = nothing
    newPos::Union{Nothing, Vector{Int32}} = nothing
    moveType::Union{Nothing, Int32} = nothing     # array of possible move types around a given bead
    tempVec::Union{Nothing, Vector{Int32}} = nothing
    nnIdxs::Union{Nothing, Vector{Int32}} = nothing

    # proposeMove! variables
    movableIdxs::Union{Nothing, Vector{Int32}} = nothing  # coordinates of unconfined particles
    freeNNs::Union{Nothing, Set{Int32}} = nothing   # move proposal probabilities
    nnsOld::Union{Nothing, Int32} = nothing          # number of nearest neighbs of old site
    nnsNew::Union{Nothing, Int32} = nothing          # number of nearest neighbs of new site
    hastingsFactor::Union{Nothing, Float64} = nothing
    
    # energy
    ham::Union{Nothing, Float64} = nothing       # Hamiltonian
    deltaH::Union{Nothing, Float64} = nothing    # Hamiltonian increment of a given move
    deltaHP::Union{Nothing, Float64} = nothing   # Hamiltonian increment from partner move
end


######## FastBulkMonomer #######
# MARK: -FastBulkMonomer

Base.@kwdef mutable struct FastBulkMonomer <: AbstractFastBulkMonomer
    
    id::Int32 = begin
      SPECIES_ID_COUNTER[] += 1
      SPECIES_ID_COUNTER[]
    end

    name::String = ""    # unique name of this species

    # required fields
    length::Int32        # number of beads

    chemPot::Union{Nothing,Float64} = nothing   # monomer chemical potential
    intPot::Union{Nothing,Float64} = nothing   # monomer interaction potential

    # key vector, position and initial, final bead positions
    boxlength::Union{Nothing, Int32} = nothing
    keys_vec::Union{Nothing, Vector{Int32}} = nothing  # pre-allocated key vector
    pos::Union{Nothing, OrderedDict{Int32,Vector{Int32}}} = nothing      # xyz coordinates of all the beads
    initPos::Union{Nothing, Vector{Int32}} = nothing     # anchor position on one side
    initShape::Union{Nothing, String} = nothing
    initParams::Union{Nothing, Int32} = nothing

    # lattice, ghost and partner species
    latS::Union{Nothing, AbstractArray{Int32}} = nothing
    ghost::Union{Nothing, AbstractSpecies} = nothing

    # multi-species interaction fields
    intSpecies::Union{Nothing, Vector{Tuple{AbstractMonomer, Int32}}} = nothing  # list of (species, canonical index) pairs
    intMat::Union{Nothing, Matrix{Float64}} = nothing  # interaction matrix (symmetric, diagonal = self intPot)
    intIdx::Union{Nothing, Int32} = nothing  # this species' canonical index in intMat

    # moveTypes! variables
    bi::Union{Nothing, AbstractArray{Int32}} = nothing
    moveIdx::Union{Nothing, Int32} = nothing 
    moveIdxNew::Union{Nothing, Int32} = nothing 
    moveVec::Union{Nothing, Vector{Int32}} = nothing
    newPos::Union{Nothing, Vector{Int32}} = nothing
    moveType::Union{Nothing, Int32} = nothing     # array of possible move types around a given bead
    tempVec::Union{Nothing, Vector{Int32}} = nothing
    nnIdxs::Union{Nothing, Vector{Int32}} = nothing

    # proposeMove! variables
    movableIdxs::Union{Nothing, Set{Int32}} = nothing  # coordinates of unconfined particles
    freeNNs::Union{Nothing, Set{Int32}} = nothing   # move proposal probabilities
    nnsOld::Union{Nothing, Int32} = nothing          # number of nearest neighbs of old site
    nnsNew::Union{Nothing, Int32} = nothing          # number of nearest neighbs of new site
    hastingsFactor::Union{Nothing, Float64} = nothing
    
    # energy
    ham::Union{Nothing, Float64} = nothing       # Hamiltonian
    deltaH::Union{Nothing, Float64} = nothing    # Hamiltonian increment of a given move
    deltaHP::Union{Nothing, Float64} = nothing   # Hamiltonian increment from partner move
end


######## Lattice #######

Base.@kwdef mutable struct Lattice
    boxlength::Int32     # lattice linear dimension
    nspecies::Int32      # number of species
    lat::Union{Nothing, Array{Int32,4}} = nothing  # lattice array
    speciesDir::Union{Nothing, Dict{Int32,Int32}} = nothing    # id-indexed locations of each species in the lattice array
end


######## Rouse polymer models ########

Base.@kwdef mutable struct RousePolymer
    # physical parameters
    boltzmannConst::Float64 = 1.0
    temperature::Float64 = 1.0
    friction::Float64 = 1.0
    kuhnLength::Float64 = 1.0
    stiffness::Float64 = 3.0           # κ = 3 kBT/b² (recomputed in init!)
    dt::Float64 = 1e-5                 # physical timestep
    nsteps::Int64 = 1000
    nbeads::Int32 = 100
    r_min::Union{Nothing, Float64} = nothing   # hard repulsion distance

    # scaling
    posScale::Union{Nothing, Float64} = nothing
    timeScale::Union{Nothing, Float64} = nothing
    dtDimless::Union{Nothing, Float64} = nothing

    # working buffers (dimensionless, N×3)
    adjacency::Union{Nothing, SymTridiagonal{Float64, Vector{Float64}}} = nothing
    pos::Union{Nothing, Matrix{Float64}} = nothing
    drift::Union{Nothing, Matrix{Float64}} = nothing
    noise::Union{Nothing, Matrix{Float64}} = nothing
    initPos::Union{Nothing, Matrix{Float64}} = nothing   # physical units

    # trajectory storage
    posTraj::Union{Nothing, Array{Float64, 3}} = nothing      # (nsteps, N, 3); only when saveMode == :full
    distanceTraj::Union{Nothing, Vector{Float64}} = nothing   # always allocated

    trackedBead1::Int32 = 1
    trackedBead2::Int32 = 100

    saveMode::Symbol = :distance_only   # :full | :distance_only | :sojourn_only
    seed::Union{Nothing, Int} = nothing

    # Optional radial pair force between trackedBead1 and trackedBead2.
    # Convention: f(d_physical) = −dU/dr of the pair potential U(d).
    # e.g. U(d) = (k/2)(d−d₀)²  ⇒  attractiveForce = d -> -k*(d - d0).
    attractiveForce::Union{Nothing, Function} = nothing

    # Sojourn-time tracking of the tracked-bead distance (physical units).
    # When both a and b are set (a ≤ b), the durations of successive sojourns in
    #   A = {distance ≤ a}   and   B = {distance ≥ b},   buffer (a, b)
    # are recorded (in integration steps) into A_intervals / B_intervals using
    # Schmitt-trigger (hysteresis) assignment. Independent of saveMode; the
    # :sojourn_only mode additionally skips allocating distanceTraj.
    a::Union{Nothing, Float64} = nothing
    b::Union{Nothing, Float64} = nothing
    A_intervals::Union{Nothing, Vector{Int}} = nothing
    B_intervals::Union{Nothing, Vector{Int}} = nothing
end

###### Surface adbsorption

Base.@kwdef struct Parameters
    nb::Float64              = 1.0
    chi_b::Float64           = 0.0
    omega_b::Float64         = 0.0
    kappa::Float64           = 1.0
    phi_inf::Float64         = 0.06
    L::Float64               = 10.0
    R::Float64               = 3.0
    eps::Float64             = 1e-10
    conserve_volume::Bool    = false
    parallel_plates::Bool    = false
end
