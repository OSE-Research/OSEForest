import pandas as pd
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def file_exact(dir, filepath):


    COLUMNS = [
        "MetadataVersion",
        "MetadataID",
        "sourceIPv4Address or sourceIPv6Address",
        "destinationIPv4Address or destinationIPv6Address",
        "sourceTransportPort",
        "destinationTransportPort",
        "protocolIdentifier",
        "octetDeltaCount",
        "packetDeltaCount",
        "postOctetDeltaCount",
        "postPacketDeltaCount",
        "flowStartSeconds",
        "flowEndSecond",
        "c2s127PacketCount",
        "c2s1024PacketCount",
        "s2c127PacketCount",
        "s2c1024PacketCount",
        "c2sPayloadByteCount",
        "s2cPayloadBytes",
        "c2sRetransferPacketCount",
        "s2cRetransferPacketCount",
        "icmpRequestEntropyElement",
        "icmpReplyEntropyElement",
        "icmpRequestPayload",
        "icmpReplyPayload",
        "applicationProtocolName",
        "applicationName",
        "applicationCategoryName",
        "applicationSubCategoryName",
        "httpMethod",
        "httpRequestHostName",
        "httpResponseCode",
        "httpContentType",
        "httpRefer",
        "httpRequestUserAgent",
        "httpRequestURL",
        "httpRequestLabelNum",
        "httpReplyLabelNum",
        "httpRequestVersion",
        "httpReplyVersion",
        "fileName",
        "fileEncrypt",
        "fileType",
        "fileSize",
        "fileMd5",
        "mailContentURL",
        "mailDate",
        "mailFrom",
        "mailTo",
        "mailCC",
        "mailBCC",
        "mailSubject",
        "mailMD5",
        "DNSReplyCode",
        "DNSQueryName",
        "DNSRequestRRType",
        "DNSRRClass",
        "DNSDelay",
        "DNSReplyTTL",
        "DNSReplyIPv4",
        "DNSReplyIPv6",
        "DNSReplyRRType",
        "ReceiveTime",
        "SrcArea",
        "DestArea",
        "SrcIPUser",
        "DestIPUser",
        "SrcGeographyLocationCountryOrRegion",
        "SrcGeographyLocationCity",
        "SrcGeographyLocationLongitude",
        "SrcGeographyLocationLatitude",
        "DestGeographyLocationCountryOrRegion",
        "DestGeographyLocationCity",
        "DestGeographyLocationLongitude",
        "DestGeographyLocationLatitude",
        "downloadMailAddress",
        "DNSReplyName",
        "fileTransmissionDirection",
        "mailReceived",
        "ReceiveDate",
        "DNSRequestPackage",
        "requestLength",
        "replyLength",
        "httpCookie",
        "httpLabelList",
        "httpSelfDefineLabel",
        "IPProtocolName",
        "DNSRequestLength",
        "DNSRequestErrorLength",
        "DNSReplyLength",
        "DNSReplyErrorLength",
        "DHCPIPMACMap",
        "SrcHostUniqueID",
        "DstHostUniqueID",
        "headerHistogram",
        "payloadHistogram",
        "tcpFlagHistogram",
        "flowFirstPktTime",
        "flowLastPktTime",
        "octetDeltaCountFromTotalLen",
        "sshVersion",
        "httpRequesHead"
        # "httpRequestBody"
        # "httpReplyHead",
        # "httpReplyBody",
        # "TcpPktsLen",
        # "TcpPktsTime",
        # "TcpPayloadBD",
        # "TlsCliCipherSuites",
        # "TlsSvrCipherSuite",
        # "TlsCliExtType",
        # "TlsSvrExtType",
        # "TlsCliKeyExchangeLen",
        # "TlsSvrKeyExchangeLen",
        # "CertDuration",
        # "CertSelfSigned",
        # "CertSAN",
        # "CertNums",
        # "CertIsCA",
        # "DnsIPCount",
        # "TlsVersion",
        # "TlsEllipticCurves",
        # "TlsECPointFmt",
        # "TlsCipherSuiteLen",
        # "TlsCompressionLength",
        # "TlsCompressionMethods",
        # "TlsSignatureAlg",
        # "TlsFingerPrint",
        # "StandBy1",
        # "StandBy2",
        # "StandBy3",
        # "StandBy4",
        # "StandBy5",
        # "StandBy6",
        # "StandBy7",
        # "StandBy8",
        # "StandBy9",
        # "StandBy10",
        # "StandBy11",
        # "StandBy12",
        # "StandBy13",
        # "StandBy14",
        # "StandBy15"
    ]
    file_dir = "../DNS_o/latest_metadata_sample/" + dir  # file directory
    all_csv_list = os.listdir(file_dir)  # get 文件 list
    for single_csv in tqdm(all_csv_list, desc='Extract files', unit='files'):
        path = os.path.join(file_dir, single_csv)
        # print(path)
        if os.path.exists(path) and os.path.getsize(path):
            single_data_frame = pd.read_table(path, sep='^', header=None, encoding='ISO-8859-1', low_memory=False)
            if single_csv == all_csv_list[0]:
                all_data_frame = single_data_frame
            else:  # concatenate all csv to a single dataframe, ingore index
                # print(single_csv)
                all_data_frame = pd.concat([all_data_frame, single_data_frame], ignore_index=True)
                if single_csv == all_csv_list[-1]:
                    all_data_frame.to_csv('/projects/DNS_o/bert_dns/prim_data/' + filepath, index=False, header=COLUMNS)


if __name__ == '__main__':
    # file_exact(dir='labeled_white', filepath='white.csv')
    file_exact(dir='labeled_black', filepath='black.csv')

