
# Add a loop to run test with a list of test_snr

for snr in 0 5 10 20
do
    echo "Test with SNR = $snr"
    python test_fp.py --query_lens=1,2,3,5 --test_snr=$snr --label=snr_$snr
done