#ifndef MUELU_ETI_4ARGUMENT_HPP
#define MUELU_ETI_4ARGUMENT_HPP

// The macro "MUELU_ETI_GROUP" must be defined prior to including this file.

// We need to define these typedefs as it is not possible to properly expand
// macros with colons in them
#if defined(HAVE_MUELU_TPETRA)
# include <TpetraCore_config.h>
# include <TpetraCore_ETIHelperMacros.h>
TPETRA_ETI_MANGLING_TYPEDEFS()
#endif
#if !defined(HAVE_MUELU_TPETRA) || !defined(HAVE_TPETRA_INST_SERIAL)
  typedef Kokkos::Compat::KokkosSerialWrapperNode Kokkos_Compat_KokkosSerialWrapperNode;
#endif

// Epetra = on, Tpetra = off
#if defined(HAVE_MUELU_EPETRA) && !defined(HAVE_MUELU_TPETRA)
  MUELU_ETI_GROUP(double,int,int,Kokkos_Compat_KokkosSerialWrapperNode)
#endif

// Epetra = on, Tpetra = on
#if defined(HAVE_MUELU_EPETRA) && defined(HAVE_MUELU_TPETRA)
  TPETRA_INSTANTIATE_SLGN_NO_ORDINAL_SCALAR(MUELU_ETI_GROUP)
# if !defined(HAVE_TPETRA_INST_SERIAL) || !defined(HAVE_TPETRA_INST_INT_INT)
  MUELU_ETI_GROUP(double,int,int,Kokkos_Compat_KokkosSerialWrapperNode)
# endif

#endif

// Epetra = off, Tpetra = on
#if !defined(HAVE_MUELU_EPETRA) && defined(HAVE_MUELU_TPETRA)
  TPETRA_INSTANTIATE_SLGN_NO_ORDINAL_SCALAR(MUELU_ETI_GROUP)
#endif

#endif //ifndef MUELU_ETI_4ARGUMENT_HPP
