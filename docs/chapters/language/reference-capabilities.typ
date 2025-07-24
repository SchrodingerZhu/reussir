#import "/book.typ": book-page

#show: book-page.with(title: "Reference Capabilities")

= Reference Capabilities

== Values

A *value* is materializable data. It can live in a struct field, a variable, a parameter, or be returned from a function.

== Capabilities

Every value in Reussir carries a *reference capability*, which determines how the underlying data is stored and accessed. There are currently six kinds of reference capabilities:

=== Normal capabilities

Normal capabilities are the most common. Values with these capabilities can be created or modified without a `region`:

- *Value (V)*
  The target object is stored *by value*. Duplicating the binding creates a fresh copy of the object.

- *Shared (S)*
  The target object is stored on the heap and managed via reference counting. It is immutable and eligible for reuse optimizations.

- *Default (D)*
  The target object has an unspecified capability. For primitive values, this is equivalent to *Value (V)*; for other objects, it is equivalent to *Shared (S)*.

=== Region capabilities

For all region capabilities, the value is heap‑allocated and accessed by reference. In what follows, when we say a value can be *modified*, we mean its pointer can be reassigned.

- *Field (FD)*
  The target object is stored in a struct field. The field can be modified if the parent struct has the *Flex (FX)* capability.

- *Flex (FX)*
  The target object has not escaped its mutable region. All *Field (FD)* values inside the struct can be reassigned.

- *Rigid (R)*
  The target object has escaped its mutable region. It becomes immutable and is managed by a specialized reference‑counting mechanism. When a *Flex (FX)* object escapes its mutable region, it is converted to *Rigid (R)*. The object then participates in a mark-sweep phase and is collected into SCCs for memory management.
