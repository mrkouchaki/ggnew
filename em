

try:
    import oracledb
except ImportError as exc:
    raise LoaderError(
        "Oracle extraction requires the 'oracledb' package."
    ) from exc

# Enable Thick mode before the first connection.
# Use the DIRECTORY containing oci.dll, not oci.dll itself.
if oracledb.is_thin_mode():
    oracledb.init_oracle_client(
        lib_dir=r"C:\Tools\oracle\instantclient"
    )

LOGGER.info(
    "Oracle driver mode=%s client_version=%s",
    "thin" if oracledb.is_thin_mode() else "thick",
    oracledb.clientversion(),
)

kwargs: dict[str, Any] = {
    "user": self.oracle_config.user,
    "password": self.oracle_config.password,
    "dsn": self.oracle_config.dsn,
}

connection = oracledb.connect(**kwargs)