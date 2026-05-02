### Semantics

There are two distinct *tasks* that the DAG can solve: 

  (Task M) **Type materialization** of instance-level class membership from sufficient conditions:
     - For some class C, with sufficient conditions S_1, ..., S_n: an instance n is a member of (type of) class C if it meets *any* of thee sufficient conditions,
 i.e., n \in union{S_1, ..., S_n} => n \in C => typeof(n,C)
       - On a semantic level: inference of n \in C from S_1, ..., S_n should correspond to a valid entailment from ABox assertions under the given DL profile (assuming prior ABox/TBox consistency).
     - This is the forward pass of the DAG: score(n,C) for some instance n is determined based on its existing types and relation to other nodes in the graph, and the DAG propagates sufficient conditions from the TBox.
       - Disjunctions in antecedents, and other similar scenarios, increase complexity and could cause intractability or incorrectness without care. This is why we are initially focusing on OWL EL.
  
  (Task A) **Admissibility testing** for instance-level class membership from necessary conditions:
     - For some class C, with necessary conditions N1, ..., Nn: an instance n is admissible as a member of (type of) class C it it meets *all* of these necessary conditions, 
 i.e., n \in intersection{N_1, ..., N_n} => admissible(n,C)
       - On a semantic level: inference of admissible(n,C) from N_1, ..., N_n should imply { "n \in C" } \cup ABox is consistent under the given DL profile (assuming prior ABox/TBox consistency).
         - Note that this does not imply that "n \in C" can be concluded, as these are only necessary conditions; rather, it implies that "n \in C" is "safe" to add as an additional assertion.
         - Note also that if non-positive or non-monotone constructs are allowed in the OWL DL profile, then in order for this consistency to hold, it must be true that "n \notin C" cannot be concluded from existing sufficient conditions in the ABox. 
     - This is the backwards pass of the DAG" score(n,C) for some instance n is determined based on its existing types and relation to other nodes in the graph, and the DAG propagates necessary conditions from the TBox.
       - Complements with sufficient or necessary conditions, and other similar scenarios, and could cause intractability or incorrectness without care. This is why we are initially focusing on OWL EL.
  
  For both Task M and Task A, we do not want to require *completeness*, though it is a nice property to have if possible under certain limited subsets of OWL DL. Instead, we want to ensure *correctness*, and completeness is a secondary priority: be as complete as possible without compromising computational efficiency and/or correctness.
  - In other words: we don't necessarily have to make every valid typeof() and admissible() assertion, and an output score of s(n,C) really means "unknown", not "untrue".
  
  In addition: Task A need not operate under strict OWA semantics. If additional assertions were added to the ABox, they could invalidate or cause inconsistency with the admissibility scores. In other words: admissibility is determined in relation to what is known, not what might be known later. This may also be a desirable property for Task M, as well; but since Task M corresponds more closely to standard OWL DL entailment semantics, it is too soon to make a jump of this nature.
  
Using these tasks, the reasoner has three distinct modes:

  (1) query: This mode checks for admissibility for the targeted class set. It should first perform Task M to a fixpoint, materializing missing type assertions, then perform Task A once, generating individual admissibility scores (not necessarily mutually compatible ones). In theory, Task M iteration to a fixpoint is only necessary if the TBox is cyclic.

  (2) stratified: This mode mimics OWA reasoning. It consists of performing {Task M + Task A} iteratively to a fixpoint: materializing type assertions, then performing Task A to check if necessary conditions are satisfied for our tentatively materialized new assertions, and blocking any unconfirmed materializations. This second phase is necessary when it is possible that inference of multiple types simulataneously may cause incompatibilities or inconsistencies. In addition, 

  (3) query-filter: This mode combines the admissibility checking of query with the OWA reasoning of stratified. Essentially, it first performs the query mode, generating tentative individual admissibility scores, and then it verifies these with repeated Task M + Task A testing. The final result should only be mutually compatible admissibility assignments: i.e., if all nodes that were admissible as these types were then to have these types asserted, it would not cause inconsistencies. In addition, all logical implications from these assertions is also computed, along with additional saturation of admissibility scores as a result.
  
This is, at least, one possible version of the semantics, which may or may not match what we have implemented. In addition, it definitely does not match naive use of a reasoner: the reasoner can be used to verify our entailments are valid, compare completeness, and check for consistency of the admissibility results; but (2) is the only mode that may be directly comparable. 