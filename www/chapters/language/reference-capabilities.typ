#import "/book.typ": book-page

#show: book-page.with(title: "Reference Capabilities")

= Background

Capabilities in `Reussir` are similar to modes in OxCaml.
It is used to determine the underlying storage of a type and its behavior
when interacting with multi-threading, projection and mutability.

== Type Capabilities

A type in `Reussir` programming language has capability, which
decides the storage and permitted operations on the type.

All primitive types have `Value (V)` capability, which means that
the type is unboxed and always being passed by value.

A record type can have one of the following capabilities:

- `Value (V)`: similar to primitive types, the record is not boxed and being
  passed/stored as values across interfaces.
- `Shared (S)`: the record is boxed and passed as `RcPtr`.
- `Regional (RE)`: the record can only be created within regional expression.
  Depending on the reference's capability, it can be mutated within region (`Flex (F)`).
  Or it can be referenced outside region (`Rigid (R)`).

== About Fields

Specific to a record with `RE` capability, member fields can be
annotated with `Field (FE)` capability, that is, the member can be mutated if it
is within the region. A member of `FE` capability must be of a record type
with `RE` capability.

A `RE` record type can also be used as a member field of value/shared record, where
it is regarded as a `Rigid` reference.

== Reference Capabilities

A reference (including `Ref/Rc`) in `Reussir` language can have
the following capability:

1. `Shared (S)`: a reference to Rc-shared data. Hence, it is immutable.
2. `Flex (F)`: a flexible reference to regional data, hence it can be used to 
   get/modify mutable fields.
3. `Rigid (R)`: a rigid reference to regional data which is already frozen.


== Changes to Original Design

1. There is no longer `Unspecified (U)` capability at the view of frontend. However,
   the backend may still have this attribute for the ease of handling situations
   where capabilities are not related.
2. A record now has fixed capability, which makes it easier to do type inference.

== Future Directions

- Describe the atomic/nonatomic axis (the `Arc` and synchronized-cell
  surface already exercises it).
- Consider adding more axes.
