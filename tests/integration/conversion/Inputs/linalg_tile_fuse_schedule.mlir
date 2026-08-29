// Tile-and-fuse schedule: tile the trailing elementwise stage into an 8x8
// scf.forall, then fuse matmul and fill into the tile loop.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %relu = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %forall = transform.structured.tile_using_forall %relu tile_sizes [8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %fused_mm, %forall2 = transform.structured.fuse_into_containing_op %mm into %forall
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill = transform.structured.match ops{["linalg.fill"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %fused_fill, %forall3 = transform.structured.fuse_into_containing_op %fill into %forall2
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
